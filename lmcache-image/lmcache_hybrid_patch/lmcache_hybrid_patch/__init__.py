# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin (entry point group ``vllm.general_plugins``).

Port of intel-accel-for-llm kvshrink/vllm_patch.py (PR #41) to
LMCacheMPConnector. With prefix caching disabled, vLLM forces
mamba_cache_mode='none' and sets mamba_block_size to max_model_len, so the KV
cache groups of a hybrid model end up with different block sizes. When
LMCacheMPConnector is configured this plugin:
- switches mamba_cache_mode to 'align' when prefix caching is disabled, so the
  platform aligns mamba_block_size to the attention block size;
- bypasses VllmConfig.validate_mamba_block_size in that case.
"""

import dataclasses

from vllm.logger import init_logger

# Under the "vllm" namespace so vLLM's log handler prints it.
logger = init_logger(f"vllm.{__name__}")


def _uses_lmcache_mp(vllm_config) -> bool:
    kv_config = vllm_config.kv_transfer_config
    return kv_config is not None and kv_config.kv_connector == "LMCacheMPConnector"


def _patch_mamba_cache_mode() -> None:
    from vllm.model_executor.models.config import MambaModelConfig

    orig = MambaModelConfig.verify_and_update_config.__func__

    def verify_and_update_config(cls, vllm_config) -> None:
        orig(cls, vllm_config)
        cache_config = vllm_config.cache_config
        if not _uses_lmcache_mp(vllm_config):
            return
        if cache_config.enable_prefix_caching:
            logger.info(
                "LMCache hybrid patch: prefix caching enabled, keeping "
                "mamba_cache_mode=%r",
                cache_config.mamba_cache_mode,
            )
            return
        # The platform then aligns mamba_block_size to the attention block size.
        logger.info(
            "LMCache hybrid patch: prefix caching disabled, overriding "
            "mamba_cache_mode %r -> 'align' (mamba_block_size=%s)",
            cache_config.mamba_cache_mode,
            cache_config.mamba_block_size,
        )
        cache_config.mamba_cache_mode = "align"

    MambaModelConfig.verify_and_update_config = classmethod(verify_and_update_config)


def _patch_mamba_block_size_validator() -> None:
    from pydantic.dataclasses import rebuild_dataclass
    from vllm.config import VllmConfig

    validators = VllmConfig.__pydantic_decorators__.model_validators
    decorator = validators["validate_mamba_block_size"]
    orig = decorator.func

    def validate_mamba_block_size(self):
        if (
            self.model_config is not None
            and _uses_lmcache_mp(self)
            and self.cache_config.mamba_cache_mode == "align"
        ):
            logger.info_once(
                "LMCache hybrid patch: skipping validate_mamba_block_size "
                "(mamba_block_size=%s, max_model_len=%d, prefix caching=%s)",
                self.cache_config.mamba_block_size,
                self.model_config.max_model_len,
                self.cache_config.enable_prefix_caching,
            )
            return self
        return orig(self)

    # Pydantic compiles validators into the schema, so rebuild it.
    validators["validate_mamba_block_size"] = dataclasses.replace(
        decorator, func=validate_mamba_block_size
    )
    rebuild_dataclass(VllmConfig, force=True)


def register() -> None:
    from vllm.model_executor.models.config import MambaModelConfig

    if getattr(MambaModelConfig, "_lmcache_hybrid_patched", False):
        return
    _patch_mamba_cache_mode()
    _patch_mamba_block_size_validator()
    MambaModelConfig._lmcache_hybrid_patched = True
    logger.info(
        "LMCache hybrid patch registered: MambaModelConfig.verify_and_update_config, "
        "VllmConfig.validate_mamba_block_size"
    )
