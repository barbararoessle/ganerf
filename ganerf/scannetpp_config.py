from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from .scannetpp_dataparser import ScannetppDataParserConfig

scannetpp_dataparser = DataParserSpecification(config=ScannetppDataParserConfig())
