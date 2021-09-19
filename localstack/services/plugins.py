import abc
import functools
import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Callable, Dict, List, Optional

import requests
from readerwriterlock import rwlock
from requests.models import Request

from localstack import config
from localstack.plugin import Plugin, PluginManager, PluginSpec
from localstack.utils.bootstrap import canonicalize_api_names
from localstack.utils.common import clone, poll_condition

# set up logger
LOG = logging.getLogger(__name__)

# namespace for AWS provider plugins
PLUGIN_NAMESPACE = "localstack.aws.provider"

# TODO: Define interfaces for ServiceLifecycle, SchemaValidation, SecurityEnforcement, etc.
#       to achieve composable ServicePlugins.
#  Ultimately, the plugin metamodel will allow to easily add new ServicePlugins that consist of:
#     - optional initializer (e.g., download dependencies, apply patches)
#     - service lifecycle (start, stop, pause)
#     - health check
#     - state manager for persistent state (potentially composite)
#     - request/schema validator (future work)
#     - security interceptor (future work)
#     ...

# maps service names to health status
STATUSES: Dict[str, Dict] = {}


# ---------------------------
# STATE SERIALIZER INTERFACE
# ---------------------------


class PersistenceContext:
    state_dir: str
    lock: rwlock.RWLockable

    def __init__(self, state_dir: str = None, lock: rwlock.RWLockable = None):
        # state dir (within DATA_DIR) of currently processed API in local file system
        self.state_dir = state_dir
        # read-write lock for concurrency control of incoming requests
        self.lock = lock


class StateSerializer(abc.ABC):
    """A state serializer encapsulates the logic of persisting and loading service state to/from disk."""

    @abc.abstractmethod
    def restore_state(self, context: PersistenceContext):
        """Restore state from the underlying persistence file"""
        pass

    @abc.abstractmethod
    def update_state(self, context: PersistenceContext, request: Request):
        """Update persistence state based on the incoming request"""
        pass

    @abc.abstractmethod
    def is_write_request(self, request: Request) -> bool:
        """Returns whether the given request is a write request that should trigger serialization"""
        return False

    def get_lock_for_request(self, request: Request) -> Optional[rwlock.Lockable]:
        """Returns a lock (or None) that should be used to guard the given request, for concurrency control"""
        return None

    def get_context(self) -> PersistenceContext:
        """Returns the current persistence context"""
        return None


class StateSerializerComposite(StateSerializer):
    """Composite state serializer that delegates the requests to a list of underlying concrete serializers"""

    def __init__(self, serializers: List[StateSerializer] = None):
        self.serializers: List[StateSerializer] = serializers or []

    def restore_state(self, context: PersistenceContext):
        for serializer in self.serializers:
            serializer.restore_state(context)

    def update_state(self, context: PersistenceContext, request: Request):
        for serializer in self.serializers:
            serializer.update_state(context, request)

    def is_write_request(self, request: Request) -> bool:
        return any(ser.is_write_request(request) for ser in self.serializers)

    def get_lock_for_request(self, request: Request) -> Optional[rwlock.Lockable]:
        if self.serializers:
            return self.serializers[0].get_lock_for_request(
                request
            )  # return lock from first serializer

    def get_context(self) -> PersistenceContext:
        if self.serializers:
            return self.serializers[0].get_context()  # return context from first serializer


# maps service names to serializers (TODO: to be encapsulated in ServicePlugin instances)
SERIALIZERS: Dict[str, StateSerializer] = {}


# -----------------
# PLUGIN UTILITIES
# -----------------


class Service(object):
    def __init__(self, name, start, check=None, listener=None, priority=0, active=False):
        self.plugin_name = name
        self.start_function = start
        self.listener = listener
        self.check_function = check
        self.priority = priority
        self.default_active = active

    def start(self, asynchronous):
        kwargs = {"asynchronous": asynchronous}
        if self.listener:
            kwargs["update_listener"] = self.listener
        return self.start_function(**kwargs)

    def check(self, expect_shutdown=False, print_error=False):
        if not self.check_function:
            return
        return self.check_function(expect_shutdown=expect_shutdown, print_error=print_error)

    def name(self):
        return self.plugin_name

    def is_enabled(self, api_names=None):
        if self.default_active:
            return True
        if api_names is None:
            api_names = canonicalize_api_names()
        return self.name() in api_names


class ServiceState(Enum):
    UNKNOWN = 0
    AVAILABLE = 1
    DISABLED = 2
    STARTING = 3
    RUNNING = 4
    # UPDATING = 5
    STOPPING = 6
    STOPPED = 7
    ERROR = 8


class ServiceContainer:
    """
    Holds a service, its state, and exposes lifecycle methods of the service.
    """
    service: Service
    state: ServiceState
    lock: threading.RLock
    errors: List[Exception]

    def __init__(self, service: Service, state=ServiceState.UNKNOWN):
        self.service = service
        self.state = state
        self.lock = threading.RLock()
        self.errors = list()

    def get(self) -> Service:
        return self.service

    def start(self) -> bool:
        try:
            self.state = ServiceState.STARTING
            self.service.start(asynchronous=True)
            self.service.check()
            self.state = ServiceState.RUNNING
            return True
        except Exception as e:
            LOG.error("error while starting service %s: %s", self.service.name())
            self.state = ServiceState.ERROR
            self.errors.append(e)
            return False

    def check(self) -> bool:
        try:
            self.service.check()
            return True
        except:
            return False

    def stop(self):
        try:
            self.state = ServiceState.STOPPING
            # TODO: actually stop the service
            self.state = ServiceState.STOPPED
        except Exception as e:
            self.state = ServiceState.ERROR
            self.errors.append(e)


class ServiceManager:
    services: Dict[str, ServiceContainer]

    def __init__(self) -> None:
        super().__init__()
        self.services = dict()
        self.running = set()

    def add_service(self, service: Service) -> bool:
        existing = self.services.get(service.name())
        if existing:
            if existing.service.priority > service.priority:  # FIXME: old concept that may not be needed anymore
                return False

        state = ServiceState.AVAILABLE if service.is_enabled() else ServiceState.DISABLED
        self.services[service.name()] = ServiceContainer(service, state)

        return True

    def list_available(self) -> List[str]:
        return list(self.services.keys())

    def exists(self, name: str) -> bool:
        return name in self.services

    def is_running(self, name: str) -> bool:
        return self.get_state(name) == ServiceState.RUNNING

    def get_state(self, name: str) -> Optional[ServiceState]:
        if not self.exists(name):
            return None
        return self.services[name].state

    def require(self, name: str) -> Service:
        """
        High level function that always returns a running service, or raises an error. If the service is in a state
        that it could be transitioned into a running state, then invoking this function will attempt that transition,
        e.g., by starting the service if it is available.
        """
        container = self.services.get(name)

        if not container:
            raise ValueError("no such service %s" % name)

        if container.state == ServiceState.STARTING:
            if not poll_condition(lambda: container.state != ServiceState.STARTING, timeout=5):
                raise TimeoutError("gave up waiting for service %s to start" % name)

        with container.lock:
            if container.state == ServiceState.RUNNING:
                return container.service

            if container.state == ServiceState.ERROR:
                # raise any capture error
                raise container.errors[-1]

            if container.state == ServiceState.AVAILABLE:
                # container
                if container.start():
                    record_service_health(name, "running")  # FIXME
                    return container.service
                else:
                    raise container.errors[-1]

        raise ValueError("service %s is not ready (%s) and could not be started" % (name, container.state))

    def get_service(self, name: str) -> Optional[Service]:
        container = self.services.get(name)
        if container:
            return container.service
        else:
            return None

    # legacy map compatibility

    def items(self):
        return {container.service.name(): container.service for container in self.services.values()}.items()

    def keys(self):
        return self.services.keys()

    def values(self):
        return [container.service for container in self.services.values()]

    def get(self, key):
        return self.get_service(key)

    def __iter__(self):
        return self.services


class ServicePlugin(Plugin):
    service: Service
    api: str

    @abc.abstractmethod
    def create_service(self) -> Service:
        raise NotImplementedError

    def load(self):
        self.service = self.create_service()
        return self.service


class ServicePluginAdapter(ServicePlugin):
    def __init__(
            self,
            api: str,
            create_service: Callable[[], Service],
            should_load: Callable[[], bool] = None,
    ) -> None:
        super().__init__()
        self.api = api
        self._create_service = create_service
        self._should_load = should_load

    def should_load(self) -> bool:
        if self._should_load:
            return self._should_load()
        return True

    def create_service(self) -> Service:
        return self._create_service()


def aws_provider(api: str = None, name="default", should_load: Callable[[], bool] = None):
    """
    Decorator for marking methods that create a Service instance as a ServicePlugin. Methods marked with this
    decorator are discoverable as a PluginSpec within the namespace "localstack.aws.provider", with the name
    "<api>:<name>". If api is not explicitly specified, then the method name is used as api name.
    """

    def wrapper(fn):
        # sugar for being able to name the function like the api
        _api = api or fn.__name__

        # this causes the plugin framework into pointing the entrypoint to the original function rather than the
        # nested factory function
        @functools.wraps(fn)
        def factory() -> ServicePluginAdapter:
            return ServicePluginAdapter(api=_api, should_load=should_load, create_service=fn)

        return PluginSpec(PLUGIN_NAMESPACE, f"{_api}:{name}", factory=factory)

    return wrapper


class ServicePluginManager(ServiceManager):
    plugin_manager: PluginManager[ServicePlugin]

    def __init__(self, plugin_manager: PluginManager[ServicePlugin] = None) -> None:
        super().__init__()
        self.plugin_manager = plugin_manager or PluginManager(PLUGIN_NAMESPACE)
        self._api_provider_specs = None

    def list_available(self) -> List[str]:
        return list(self.api_provider_specs.keys())

    @property
    def api_provider_specs(self) -> Dict[str, List[PluginSpec]]:
        """
        Returns all PluginSpecs within the service plugin namespace and parses their name according to the convention,
        that is "<api>:<provider>". The result is a dictionary that maps api => List[PluginSpec (a provider)].
        """
        if self._api_provider_specs is None:
            self._api_provider_specs = self._resolve_api_provider_specs()
        return self._api_provider_specs

    def get_service(self, name: str) -> Optional[Service]:
        service = super().get_service(name)

        if service:
            return service

        plugin = self._load_service_plugin(name)
        if not plugin:
            return None

        self.plugin_manager.is_loaded(plugin.name)

        self.add_service(service)
        return service

    def _load_service_plugin(self, name: str) -> Optional[ServicePlugin]:
        providers = self.api_provider_specs.get(name)
        if not providers:
            # no providers for this api
            return None

        if len(providers) == 1:
            provider = providers[0]
        else:
            LOG.warning("more than one provider for %s exists: %s", name, providers)
            provider = providers[0]  # TODO read preferred provider for API from settings/config

        plugin_name = f"{name}:{provider}"
        service = self.plugin_manager.load(plugin_name)
        return service

    def _resolve_api_provider_specs(self) -> Dict[str, List[PluginSpec]]:
        result = defaultdict(list)

        for spec in self.plugin_manager.list_plugin_specs():
            api, provider = spec.name.split(':')  # TODO: error handling, faulty plugins could break the runtime
            result[api].append(provider)

        return result


def register_service(service):
    SERVICE_PLUGINS.add_service(service)


# map of service plugins, mapping from service name to plugin details
SERVICE_PLUGINS: ServiceManager = ServiceManager()


# -------------------------
# HEALTH CHECK API METHODS
# -------------------------


def get_services_health(reload=False):
    if reload:
        reload_services_health()
    result = clone(dict(STATUSES))
    result.get("services", {}).pop("edge", None)
    return result


def set_services_health(data):
    status = STATUSES["services"] = STATUSES.get("services", {})
    for key, value in dict(data).items():
        parent, _, child = key.partition(":")
        if child:
            STATUSES[parent] = STATUSES.get(parent, {})
            STATUSES[parent][child] = value
            data.pop(key)
    status.update(data or {})
    return get_services_health()


# -----------------------------
# INFRASTRUCTURE HEALTH CHECKS
# -----------------------------


def check_infra(retries=10, expect_shutdown=False, apis=None, additional_checks=[]):
    try:
        apis = apis or canonicalize_api_names()
        print_error = retries <= 0

        # loop through plugins and check service status
        for name, plugin in SERVICE_PLUGINS.items():
            if name in apis:
                check_service_health(
                    api=name, print_error=print_error, expect_shutdown=expect_shutdown
                )

        for additional in additional_checks:
            additional(expect_shutdown=expect_shutdown)
    except Exception as e:
        if retries <= 0:
            LOG.exception("Error checking state of local environment (after some retries)")
            raise e
        time.sleep(3)
        check_infra(
            retries - 1,
            expect_shutdown=expect_shutdown,
            apis=apis,
            additional_checks=additional_checks,
        )


def wait_for_infra_shutdown(apis=None):
    apis = apis or canonicalize_api_names()

    names = [name for name, plugin in SERVICE_PLUGINS.items() if name in apis]

    def check(name):
        check_service_health(api=name, expect_shutdown=True)
        LOG.debug("[shutdown] api %s has shut down", name)

    # no special significance to 10 workers, seems like a reasonable number given the number of services we have
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(check, names)


def check_service_health(api, print_error=False, expect_shutdown=False):
    try:
        plugin = SERVICE_PLUGINS.get(api)
        plugin.check(expect_shutdown=expect_shutdown, print_error=print_error)
        record_service_health(api, "running")
    except Exception as e:
        if not expect_shutdown:
            LOG.warning('Service "%s" not yet available, retrying...' % api)
        else:
            LOG.warning('Service "%s" still shutting down, retrying...' % api)
        raise e


def reload_services_health():
    check_infra(retries=0)


def record_service_health(api, status):
    # TODO: consider making in-memory calls here, to optimize performance
    data = {api: status}
    health_url = "%s/health" % config.get_edge_url()
    try:
        requests.put(health_url, data=json.dumps(data), verify=False)
    except Exception:
        # ignore for now, if the service is not running
        pass
