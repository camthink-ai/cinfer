import { useRequest } from 'ahooks';

import { dashboardAPI, getResponseData, awaitWrap, isRequestSuccess } from '@/services/http';

/**
 * System info data
 */
export function useSystemInfo() {
    const { data: systemInfo } = useRequest(
        async () => {
            const [error, resp] = await awaitWrap(dashboardAPI.getSystemInfo());
            if (error || !isRequestSuccess(resp)) {
                return;
            }

            return getResponseData(resp);
        },
        {
            manual: false,
        },
    );

    return {
        systemInfo,
        modelsStatus: systemInfo?.models_stats,
    };
}
