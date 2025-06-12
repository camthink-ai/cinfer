import { useRequest, useMemoizedFn } from 'ahooks';
import dayjs from 'dayjs';

import { DEFAULT_DATA_TIME_FORMAT } from '@milesight/shared/src/services/time';

import {
    dashboardAPI,
    getResponseData,
    awaitWrap,
    isRequestSuccess,
    type SystemMetricsType,
} from '@/services/http';

/**
 * System Metrics data
 * CPU/GPU/Memory
 */
export function useSystemMetrics() {
    /**
     * transform data into echarts display data format
     */
    const getStatisticsData = useMemoizedFn(
        (type: keyof SystemMetricsType, data: SystemMetricsType[]) => {
            return data
                .filter(item => !Number.isNaN(Number(item[type])))
                .map(item => [
                    dayjs(item.timestamp).format(DEFAULT_DATA_TIME_FORMAT.fullDateTimeSecondFormat),
                    item[type],
                ]);
        },
    );

    const { data } = useRequest(
        async () => {
            const [error, resp] = await awaitWrap(dashboardAPI.getSystemMetrics());
            if (error || !isRequestSuccess(resp)) {
                return;
            }

            const data = getResponseData(resp);
            if (!Array.isArray(data)) {
                return data;
            }

            return {
                cpuData: getStatisticsData('cpu_usage', data),
                gpuData: getStatisticsData('gpu_usage', data),
                memoryData: getStatisticsData('mem_usage', data),
            };
        },
        {
            manual: false,
        },
    );

    return {
        /** cpu statistics data */
        cpuData: data?.cpuData || [],
        /** gpuData statistics data */
        gpuData: data?.gpuData || [],
        /** memoryData statistics data */
        memoryData: data?.memoryData || [],
    };
}
