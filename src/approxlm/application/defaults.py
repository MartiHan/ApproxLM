from __future__ import annotations
import re
DEFAULT_MODEL='qanastek/XLMRoberta-Alexa-Intents-Classification'
DEFAULT_DATASET='AmazonScience/massive'
DEFAULT_DECODER_MODEL='Qwen/Qwen2-0.5B'
DEFAULT_DECODER_DATASET='wikitext'
APPROX_OPTIONS=['fp32','int8_exact','mul8s_exact','mul8s_exact2','mul8s_1KVA','mul8s_1KVB','mul8s_1KR6','mul8s_1KR6_swapped','mul8s_1KR3','mul8s_1KVM','mul8s_1KVM_swapped']
DEFAULT_TRACE_ENABLED=True
DEFAULT_ATTENTION_MODE='cls_row'
BLOCK_INDEX_PATTERN=re.compile(r'^(.*?)(?:\.)(\d+)(?:\.)(.+)$')
