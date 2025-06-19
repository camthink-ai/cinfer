import { useRequest } from 'ahooks';
import { awaitWrap, isRequestSuccess, modelAPI, getResponseData } from '@/services/http';

export function useGetModels() {
    const { data: modelOptions, loading: modelsLoading } = useRequest(async () => {
        const [err, resp] = await awaitWrap(
            modelAPI.getModelList({
                page: 1,
                page_size: 9999,
            }),
        );
        if (err || !isRequestSuccess(resp)) {
            return;
        }

        const data = getResponseData(resp);
        return (data || []).map(m => ({ label: m.name, value: m.id }));
    });

    return {
        modelOptions,
        modelsLoading,
    };
}
