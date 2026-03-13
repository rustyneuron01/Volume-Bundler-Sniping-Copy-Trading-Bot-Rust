import functools
from typing import Optional

import bittensor as bt

from gas.types import DiscriminatorModelId as ModelId, DiscriminatorModelMetadata as ModelMetadata
from gas.utils import run_in_thread


class ChainModelMetadataStore:
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        netuid: int,
    ):
        self.subtensor = subtensor
        self.netuid = netuid

    async def store_model_metadata(
        self,
        wallet: str,
        model_id: ModelId,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        ttl: int = 60,
    ):
        """Stores model metadata on this subnet for a specific wallet."""
        data = model_id.to_compressed_str()

        commit_partial = functools.partial(
            self.subtensor.commit,
            wallet,
            self.netuid,
            data,
        )

        run_in_thread(commit_partial, ttl)

    async def retrieve_model_metadata(
        self, uid: int, hotkey: str, ttl: int = 60
    ) -> Optional[ModelMetadata]:
        """Retrieves model metadata on this subnet for specific hotkey"""

        metadata_partial = functools.partial(
            bt.core.extrinsics.serving.get_metadata,
            self.subtensor,
            self.netuid,
            hotkey,
        )

        commitment_partial = functools.partial(
            self.subtensor.get_commitment,
            self.netuid,
            uid,
        )

        metadata = run_in_thread(metadata_partial, ttl)

        if not metadata:
            return None

        chain_str = run_in_thread(commitment_partial, ttl)

        model_id = None
        bt.logging.info(f"chain_str: {chain_str}")
        try:
            model_id = ModelId.from_compressed_str(chain_str)
        except:
            bt.logging.trace(
                f"Failed to parse the metadata on the chain for hotkey {hotkey}."
            )
            return None

        model_metadata = ModelMetadata(id=model_id, block=metadata["block"])

        return model_metadata
