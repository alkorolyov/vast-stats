


# Define URL's and timeouts
VAST_EXPORTER_BASEURL = 'https://500.farm/vastai-exporter'
VAST_EXPORTER_TIMEOUT = 5
VAST_API_BASEURL = 'https://console.vast.ai'
VAST_API_TIMEOUT = 25   # vast-api takes longer to pop response

RETRY_TIMEOUT = 20      # timeout between failed retries
TIMEOUT = 53            # main cycle timeout. `53` is average for '500.farm' source

# Define logging options
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
MAX_LOGSIZE = 10 * 1024 * 1024  # 10Mb
LOG_COUNT = 5

VERIFIED_ENUM = {
    'unverified': 0,
    'verified': 1,
    'deverified': 2,
    'de-verified': 3
}

# Define column names and types for incoming data
INT_COLS = ['has_avx', 'bw_nvlink', 'cpu_cores', 'cpu_ram', 'hosting_type', 'disk_space',
             'dlperf', 'score', 'verification', 'reliability',
             'dph_base', 'storage_cost', 'inet_up_cost', 'inet_down_cost', 'min_bid', 'credit_discount_max',
             'total_flops', 'disk_bw', 'gpu_mem_bw', 'inet_down',
             'inet_up', 'hosting_type', 'pcie_bw', 'rented', 'static_ip',
             'compute_cap', 'direct_port_count', 'end_date',
             'gpu_display_active', 'gpu_lanes', 'gpu_ram', 'host_id',
             'machine_id', 'id', 'min_chunk', 'num_gpus', 'pci_gen',
             'num_gpus_rented', 'timestamp', 'dph_base',
            ]

STR_COLS = ['cpu_name', 'cuda_max_good', 'disk_name', 'driver_version',
            'gpu_name', 'mobo_name', 'public_ipaddr', 'country', 'isp']

FLOAT_COLS = []

# DROP_COLS = ['credit_balance', 'credit_discount', 'location', 'geolocation', 'bundle_id',
#              'discount_rate', 'discounted_dph_total', 'discounted_hourly',
#              'dlperf_per_dphtotal', 'duration', 'flops_per_dphtotal', 'start_date',
#              'verified', 'host_run_time', 'cpu_cores_effective', 'gpu_frac', 'chunks']


# Group columns
AVG_COLS = ['disk_bw', 'gpu_mem_bw', 'pcie_bw',
            'dlperf', 'inet_down', 'inet_up',
            'score']

HARDWARE_COLS = ['compute_cap', 'total_flops',
                 'cpu_cores', 'cpu_name', 'has_avx',
                 'disk_name', 'hosting_type', 'mobo_name',
                 'gpu_name', 'num_gpus', 'pci_gen', 'gpu_lanes', 'gpu_ram', 'bw_nvlink']

EOD_COLS = ['cuda_max_good', 'driver_version', 'direct_port_count',
            'min_chunk', 'verification', 'end_date',
            'country', 'isp', 'public_ipaddr', 'static_ip']

COST_COLS = ['dph_base', 'storage_cost', 'inet_up_cost',
             'inet_down_cost', 'min_bid', 'credit_discount_max']


# Columns to keep
ID_COLS = ['id', 'host_id', 'machine_id']
SINGLE_COLS = ['timestamp', 'cpu_ram', 'disk_space', 'reliability', 'num_gpus_rented', 'rented']

KEEP_COLS = (AVG_COLS +
             HARDWARE_COLS +
             EOD_COLS +
             COST_COLS +
             ID_COLS +
             SINGLE_COLS)

