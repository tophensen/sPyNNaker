from abc import ABCMeta
from six import add_metaclass
from abc import abstractmethod
from spinnman.model.iptag.iptag import IPTag


@add_metaclass(ABCMeta)
class AbstractReverseIPTagableVertex(object):

    def __init__(self, tag, port, address):
        self._tag = tag
        self._port = port
        self._address = address

    def get_reverse_ip_tag(self):
        return IPTag(
            tag=self._tag, port=self._port, address=self._address)

    def set_reverse_iptag_tag(self, new_tag):
        self._tag = new_tag

    @abstractmethod
    def is_reverse_ip_tagable_vertex(self):
        pass

    @property
    def tag(self):
        return self._tag

    @property
    def port(self):
        return self._port

    @property
    def address(self):
        return self._address

    @tag.setter
    def tag(self, value):
        self._tag = value

    @port.setter
    def port(self, value):
        self._port = value

    @address.setter
    def address(self, value):
        self._address = value