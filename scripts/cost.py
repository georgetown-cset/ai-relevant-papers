"""Estimate the nominal cost for a Dataflow job.
"""
import argparse
import json
import subprocess

SEC_PER_HOUR = 60 * 60
MB_PER_GB = 1024
COST = {
    'TotalVcpuTime': .056 / SEC_PER_HOUR,
    'TotalMemoryUsage': .003557 / SEC_PER_HOUR / MB_PER_GB,
    'TotalPdUsage': .000054 / SEC_PER_HOUR,
    'TotalSsdUsage': .000298 / SEC_PER_HOUR,
    'TotalStreamingDataProcessed': .011,
}


def fetch_metrics(job_id):
    """
    Output from gcloud is an array with elements like::

        {'name': {'context': {'original_name': 'Service-cpu_num'}, 'name': 'CurrentVcpuCount', 'origin': 'dataflow/v1b3'}, 'scalar': 152, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-mem_mb'}, 'name': 'CurrentMemoryUsage', 'origin': 'dataflow/v1b3'}, 'scalar': 1011712, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-pd_gb'}, 'name': 'CurrentPdUsage', 'origin': 'dataflow/v1b3'}, 'scalar': 3800, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-pd_ssd_gb'}, 'name': 'CurrentSsdUsage', 'origin': 'dataflow/v1b3'}, 'scalar': 0, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-pd_gb_seconds'}, 'name': 'TotalPdUsage', 'origin': 'dataflow/v1b3'}, 'scalar': 80945756, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-cpu_num_seconds'}, 'name': 'TotalVcpuTime', 'origin': 'dataflow/v1b3'}, 'scalar': 3237830, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-mem_mb_seconds'}, 'name': 'TotalMemoryUsage', 'origin': 'dataflow/v1b3'}, 'scalar': 21550998323, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-pd_ssd_gb_seconds'}, 'name': 'TotalSsdUsage', 'origin': 'dataflow/v1b3'}, 'scalar': 0, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-streaming_service_gb'}, 'name': 'TotalStreamingDataProcessed', 'origin': 'dataflow/v1b3'}, 'scalar': 0, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-shuffle_service_actual_gb'}, 'name': 'TotalShuffleDataProcessed', 'origin': 'dataflow/v1b3'}, 'scalar': 0, 'updateTime': '2020-02-04T12:27:24.021Z'}
        {'name': {'context': {'original_name': 'Service-shuffle_service_chargeable_gb'}, 'name': 'BillableShuffleDataProcessed', 'origin': 'dataflow/v1b3'}, 'scalar': 0, 'updateTime': '2020-02-04T12:27:24.021Z'}
    """
    args = 'gcloud beta --format json dataflow metrics list --region us-east1'.split(' ')
    args.append(job_id)
    result = subprocess.run(args, capture_output=True)
    if result.stderr:
        raise ValueError(result.stderr)
    try:
        output = json.loads(result.stdout)
    except json.decoder.JSONDecodeError as e:
        print('Ran: {}'.format(' '.join(args)))
        print('Result:')
        print(result.stdout)
        raise e
    return output


def filter_metrics(output):
    """Filter by metric name."""
    for item in output:
        name = item.get('name', {}).get('name', '')
        for k, v in COST.items():
            if k in name:
                yield name, item.get('scalar')


def calculate_cost(metrics):
    subtotals = {k: COST[k] * v for k, v in metrics}
    total = sum(v for k, v in subtotals.items())
    print(f'${total:,.0f}')


if __name__ == '__main__':
    """
    Invoke like so, specifying a Dataflow job ID:
    
        python cost.py '2020-02-04_01_44_59-5481545344696865901'
    """
    parser = argparse.ArgumentParser(description='Calculate job cost')
    parser.add_argument('job_id', help='Dataflow job ID')
    args = parser.parse_args()

    _output = fetch_metrics(args.job_id)
    _metrics = filter_metrics(_output)
    calculate_cost(_metrics)
