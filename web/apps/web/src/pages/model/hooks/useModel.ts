import { useRequest } from 'ahooks';

import { modelAPI, isRequestSuccess, awaitWrap, getResponseData } from '@/services/http';
import { useGlobalStore } from '@/stores';

export default function useModel() {
    const { setInferEngines } = useGlobalStore();

    const { run: getInferEngines } = useRequest(async () => {
        const [err, resp] = await awaitWrap(modelAPI.getEngines());
        if (err || !isRequestSuccess(resp)) {
            return;
        }

        const data = getResponseData(resp);
        setInferEngines(data);
    });

    return {
        getInferEngines,
    };
}
