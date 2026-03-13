import argparse
from pathlib import Path
from typing import Callable, List
import bittensor as bt
import copy
import inspect
import traceback
import asyncio

from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
import signal

from gas import (
    __spec_version__,
    __version__,
)
from gas.utils.metagraph import SubstrateConnectionManager
from gas.types import NeuronType
from gas.utils import ExitContext, on_block_interval
from gas.config import (
    add_args,
    add_miner_args,
    add_validator_args,
    validate_config_and_neuron_path,
)


class BaseNeuron:
    """
    Base neuron class with async substrate support and automatic reconnection.
    Provides clean async/await coordination throughout the application.
    """
    config: "bt.config"
    neuron_type: NeuronType
    exit_context = ExitContext()
    next_sync_block = None
    block_callbacks: List[Callable] = []
    substrate_manager: SubstrateConnectionManager = None
    substrate_task = None

    def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    @on_block_interval("epoch_length")
    async def maybe_sync_metagraph(self, block):
        self.check_registered()
        bt.logging.info("Resyncing Metagraph")
        self.metagraph.sync(subtensor=self.subtensor)

    async def run_callbacks(self, block):
        if (
            hasattr(self, "initialization_complete")
            and not self.initialization_complete
        ):
            bt.logging.debug(
                f"Skipping callbacks at block {block} during initialization"
            )
            return

        for callback in self.block_callbacks:
            try:
                res = callback(block)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                bt.logging.error(
                    f"Failed running callback {callback.__name__}: {str(e)}"
                )
                bt.logging.error(traceback.format_exc())

    def __init__(self, config=None):
        bt.logging.info(
            f"Bittensor Version: {bt.__version__} | SN34 Version {__spec_version__}"
        )

        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        add_args(parser)

        if self.neuron_type == NeuronType.VALIDATOR:
            bt.axon.add_args(parser)
            add_validator_args(parser)
        if self.neuron_type == NeuronType.MINER:
            bt.axon.add_args(parser)
            add_miner_args(parser)

        self.config = bt.config(parser)
        if config:
            base_config = copy.deepcopy(config)
            self.config.merge(base_config)

        if hasattr(self.config, 'cache') and hasattr(self.config.cache, 'base_dir'):
            self.config.cache.base_dir = str(Path(self.config.cache.base_dir).expanduser())

        validate_config_and_neuron_path(self.config)

        ## Add kill signals
        signal.signal(signal.SIGINT, self.exit_context.startExit)
        signal.signal(signal.SIGTERM, self.exit_context.startExit)

        ## LOGGING
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.set_info()
        if self.config.logging.debug:
            bt.logging.set_debug(True)
        if self.config.logging.trace:
            bt.logging.set_trace(True)

        ## BITTENSOR INITIALIZATION
        bt.logging.success(self.config)
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(
            config=self.config, network=self.config.subtensor.chain_endpoint
        )
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        self.loop = asyncio.get_event_loop()
        bt.logging.debug(f"Wallet: {self.wallet}")
        bt.logging.debug(f"Subtensor: {self.subtensor}")
        bt.logging.debug(f"Metagraph: {self.metagraph}")

        ## CHECK IF REGG'D
        self.check_registered()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.block_callbacks.append(self.maybe_sync_metagraph)
        self._init_substrate()

    def _init_substrate(self):
        """Initialize substrate connection manager - task will be started when event loop is running."""
        self.substrate_manager = SubstrateConnectionManager(
            url=self.config.subtensor.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY
        )
        self.substrate_task = None  # Will be created when event loop is running
        bt.logging.info("Substrate connection manager initialized (task will start when event loop runs)")

    async def start_substrate_subscription(self):
        """Start the async substrate subscription - call this from async context."""
        if self.substrate_task is None:
            self.substrate_task = self.substrate_manager.start_subscription_task(self.run_callbacks)
            bt.logging.info("🚀 Substrate subscription started")

    def check_substrate_connection(self):
        """Check substrate connection health and restart if needed."""
        # Only check if task has been created (i.e., we're in async context)
        if self.substrate_task is not None and self.substrate_task.done():
            bt.logging.info("Substrate connection lost, restarting...")
            try:
                self.substrate_task = self.substrate_manager.start_subscription_task(self.run_callbacks)
                bt.logging.info("Substrate connection restarted")
            except Exception as e:
                bt.logging.error(f"Failed to restart substrate task: {e}")
                raise

    async def shutdown_substrate(self):
        """Clean shutdown of substrate connection."""
        bt.logging.info("Shutting down substrate connection...")

        if hasattr(self, 'substrate_manager') and self.substrate_manager:
            self.substrate_manager.stop()

        if hasattr(self, 'substrate_task') and self.substrate_task and not self.substrate_task.done():
            self.substrate_task.cancel()
            try:
                await self.substrate_task
            except asyncio.CancelledError:
                pass

        bt.logging.info("Substrate shutdown complete")
