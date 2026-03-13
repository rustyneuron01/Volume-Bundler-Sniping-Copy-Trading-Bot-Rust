from typing import Dict, List, Optional, Any
import time

import bittensor as bt
import aiohttp
import asyncio

from gas.protocol.validator_requests import get_miner_type
from gas.types import MinerType


class MinerTypeTracker:
    """Tracks miner types and manages periodic updates"""

    TYPE_CACHE_TTL = 300  # Only re-query miners every 5 minutes

    def __init__(self, config, wallet, metagraph, subtensor):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor

        self.miner_types: Dict[int, MinerType] = {}
        self.last_update: Dict[int, float] = {}
        self._last_full_update: float = 0

    async def initialize_miner_types(self):
        """Initialize miner types for all registered miners"""
        bt.logging.debug("Initializing miner types for all registered miners...")
        all_miners = list(range(self.metagraph.n.item()))
        await self.update_miner_types(all_miners)
        bt.logging.debug(f"Initialized miner types for {len(self.miner_types)} miners")

    async def update_miner_types(self, miner_uids: Optional[List[int]] = None, force: bool = False):
        """Update miner types for specified miners, skipping recently queried ones"""
        current_time = time.time()

        if miner_uids is None:
            # Skip full update if we did one recently
            if not force and (current_time - self._last_full_update) < self.TYPE_CACHE_TTL:
                bt.logging.trace("Skipping miner type update - cache still fresh")
                return
            miner_uids = list(range(self.metagraph.n))
            self._last_full_update = current_time

        # Filter to only stale entries (not queried within TTL)
        if not force:
            stale_uids = [
                uid for uid in miner_uids 
                if (current_time - self.last_update.get(uid, 0)) >= self.TYPE_CACHE_TTL
            ]
            if not stale_uids:
                bt.logging.trace(f"All {len(miner_uids)} miners have fresh type cache")
                return
            miner_uids = stale_uids

        bt.logging.debug(f"Querying miner types for {len(miner_uids)} UIDs")
        async with aiohttp.ClientSession() as session:
            responses = await asyncio.gather(
                *[
                    get_miner_type(
                        uid,
                        self.metagraph.axons[uid],
                        session,
                        self.wallet.hotkey,
                        self.config.neuron.miner_total_timeout,
                    )
                    for uid in miner_uids
                ],
                return_exceptions=True,
            )

        current_time = time.time()
        for uid, response in zip(miner_uids, responses):
            # We assume a miner is a discriminator (which does not run any endpoint/hardware)
            # until we successfully get a response from a generator get_miner_info endpoint
            miner_type = MinerType.DISCRIMINATOR

            if isinstance(response, Exception):
                continue

            miner_type_str = response.get("miner_type", "") or ""
            if miner_type_str.lower() == "generator":
                miner_type = MinerType.GENERATOR

            if miner_type:
                old_type = self.miner_types.get(uid)
                self.miner_types[uid] = miner_type
                self.last_update[uid] = current_time

                if old_type != miner_type:
                    bt.logging.trace(f"UID {uid}: {old_type} -> {miner_type}")
                else:
                    bt.logging.trace(f"UID {uid}: {miner_type}")

    def get_miner_type(self, uid: int) -> Optional[MinerType]:
        return self.miner_types.get(uid)

    def get_miners_by_type(self, miner_type: MinerType) -> List[int]:
        return [uid for uid, mt in self.miner_types.items() if mt == miner_type]
