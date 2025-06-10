import { client, attachAPI, API_PREFIX } from './client';

export interface SystemMetricsType {
    timestamp: string;
    cpu_usage: number;
    mem_usage: number;
    gpu_usage: number;
}

export interface DashboardAPISchema extends APISchema {
    /** System Metrics */
    getSystemMetrics: {
        request: void;
        response: SystemMetricsType[];
    };
    /** System info */
    getSystemInfo: {
        request: void;
        response: {
            system_name: string;
            software_name: string;
            software_version: string;
            hardware_acceleration: string[];
            models_stats: {
                published_count: number;
                unpublished_count: number;
                total_count: number;
            };
            access_tokens_stats: {
                total_count: number;
                active_count: number;
            };
            os_info: {
                os_name: string;
                os_version: string;
            };
        };
    };
}

/**
 * Dashboard API services
 */
export default attachAPI<DashboardAPISchema>(client, {
    apis: {
        getSystemMetrics: `GET ${API_PREFIX}/system/metrics`,
        getSystemInfo: `GET ${API_PREFIX}/system/info`,
    },
});
