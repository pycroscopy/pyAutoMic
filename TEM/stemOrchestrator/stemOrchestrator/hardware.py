# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>

## based on the CEOS - json rpc client dummy codes
import json
import logging
from twisted.internet import reactor, defer
from twisted.protocols.basic import NetstringReceiver
from twisted.internet.protocol import ReconnectingClientFactory

logging.basicConfig()
log = logging.getLogger("CEOSAcquisition")
log.setLevel(logging.INFO)


class _CEOSProtocol(NetstringReceiver):
    def __init__(self):
        self._next_id = 1
        self._pending = {}

    def connectionMade(self):
        log.info("Connected to CEOS server")
        self.factory.client._protocol = self

    def connectionLost(self, reason):
        log.warning("Disconnected: %s", reason.getErrorMessage())
        for d in self._pending.values():
            d.errback(reason)
        self._pending.clear()

    def stringReceived(self, data):
        msg = json.loads(data)
        d = self._pending.pop(msg["id"], None)
        if d:
            if "error" in msg:
                d.errback(Exception(msg["error"]["message"]))
            else:
                d.callback(msg["result"])

    def call(self, method, params=None):
        if params is None:
            params = {}
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        d = defer.Deferred()
        self._pending[self._next_id] = d
        self.sendString(json.dumps(msg).encode("utf-8"))
        self._next_id += 1
        return d


class _CEOSFactory(ReconnectingClientFactory):
    protocol = _CEOSProtocol

    def __init__(self, client):
        self.client = client
