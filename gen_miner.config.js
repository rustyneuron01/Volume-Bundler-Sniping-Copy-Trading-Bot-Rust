const path = require('path');
const os = require('os');

// Helper functions
function getPythonInterpreter() {
  const projectRoot = __dirname;
  const venvPython = path.join(projectRoot, '.venv', 'bin', 'python');
  const fs = require('fs');
  return fs.existsSync(venvPython) ? venvPython : 'python3';
}

function getNetworkSettings(chainEndpoint) {
  if (chainEndpoint.includes('test')) return 379;
  if (chainEndpoint.includes('finney')) return 34;
  return null;
}

function getLogParam(loglevel) {
  switch (loglevel) {
    case 'trace': return '--logging.trace';
    case 'debug': return '--logging.debug';
    default: return '--logging.info';
  }
}

function getAutoUpdateParam(autoUpdate) {
  return autoUpdate === 'true' ? '' : '--autoupdate-off';
}

// Dynamically load environment variables from .env.gen_miner
const envPath = path.resolve(__dirname, '.env.gen_miner');
const envConfig = require('dotenv').config({ path: envPath });
const envFileVars = envConfig.parsed || {};

// Apply to process.env for compatibility with existing config logic
Object.assign(process.env, envFileVars);
const config = {
  // Wallet
  walletName: process.env.BT_WALLET_NAME || 'default',
  walletHotkey: process.env.BT_WALLET_HOTKEY || 'default',
  
  // Network
  chainEndpoint: process.env.BT_CHAIN_ENDPOINT || 'wss://test.finney.opentensor.ai:443',
  netuid: process.env.BT_NETUID || '379',
  
  // Axon Configuration
  axonPort: process.env.BT_AXON_PORT || '8093',
  axonIp: process.env.BT_AXON_IP || '0.0.0.0',
  axonExternalIp: process.env.BT_AXON_EXTERNAL_IP || 'auto',
  
  // Device
  device: process.env.MINER_DEVICE || 'auto',
  
  // Logging
  loglevel: process.env.BT_LOGGING_LEVEL || 'info',
  
  // Features
  autoUpdate: process.env.AUTO_UPDATE || 'false',
  
  // Miner-specific configuration
  outputDir: process.env.MINER_OUTPUT_DIR || '/tmp/generated_content',
  maxConcurrentTasks: process.env.MINER_MAX_CONCURRENT_TASKS || '5',
  workerThreads: process.env.MINER_WORKER_THREADS || '2',
  taskTimeout: process.env.MINER_TASK_TIMEOUT || '300',
  
  // Force permit setting  
  noForceValidatorPermit: process.env.MINER_NO_FORCE_VALIDATOR_PERMIT === 'true',
};

// Determine netuid (override from env or derive from chain endpoint)
const netuid = process.env.BT_NETUID || getNetworkSettings(config.chainEndpoint);

// Build command parameters
const logParam = getLogParam(config.loglevel);
const autoUpdateParam = getAutoUpdateParam(config.autoUpdate);
const pythonInterpreter = getPythonInterpreter();

// Project paths
const projectRoot = __dirname;
const minerScript = path.join(projectRoot, 'neurons', 'generator', 'miner.py');

// Allow optional override of HF cache dir via env. Must be resolved before any Python starts.
const HF_HOME_RESOLVED = process.env.HF_HOME
  || process.env.HUGGINGFACE_HOME
  || process.env.HUGGINGFACE_CACHE_DIR
  || path.join(os.homedir(), '.cache', 'huggingface');

// Build dynamic environment from .env.gen_miner file
// Adding new API keys to .env.gen_miner makes them available
// to the miner without needing to modify this config file
const DYNAMIC_ENV = {
  HF_HOME: HF_HOME_RESOLVED,
  HF_HUB_DISABLE_TELEMETRY: '1',
  ...envFileVars,
};

// Build miner arguments
const minerArgs = [
  '--wallet.name', config.walletName,
  '--wallet.hotkey', config.walletHotkey,
  '--netuid', netuid.toString(),
  '--subtensor.chain_endpoint', config.chainEndpoint,
  '--axon.port', config.axonPort,
  '--axon.ip', config.axonIp,
  '--device', config.device,
  '--miner.output-dir', config.outputDir,
  '--miner.max-concurrent-tasks', config.maxConcurrentTasks,
  '--miner.worker-threads', config.workerThreads,
  '--miner.task-timeout', config.taskTimeout,
  logParam,
];

// Add conditional arguments
if (autoUpdateParam) {
  minerArgs.push(autoUpdateParam);
}

if (config.noForceValidatorPermit) {
  minerArgs.push('--no-force-validator-permit');
}

if (config.axonExternalIp && config.axonExternalIp !== 'auto') {
  minerArgs.push('--axon.external_ip', config.axonExternalIp);
}

// PM2 Apps configuration
const apps = [
  {
    name: 'bitmind-generative-miner',
    script: minerScript,
    interpreter: pythonInterpreter,
    args: minerArgs.join(' '),
    env: {
      ...DYNAMIC_ENV,
    },
    watch: false,
    instances: 1,
    autorestart: true,
    max_restarts: 10,
    min_uptime: '10s',
    restart_delay: 4000,
    time: true,
  }
];

module.exports = {
  apps,
};

