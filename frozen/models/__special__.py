import time
import logging as log

timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
log.basicConfig(level=log.INFO)
logging = log.getLogger(__name__)
logging.info(f'[+] Logging initialized at f{timestamp}.')
HASTE_VALUE = 1
BATCH_SIZE = 1
EXECUTION_BS = 512
