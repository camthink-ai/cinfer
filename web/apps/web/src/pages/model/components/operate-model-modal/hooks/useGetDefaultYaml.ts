import { useRequest } from 'ahooks';
import { modelAPI, getResponseData, isRequestSuccess, awaitWrap } from '@/services/http';

export function useGetDefaultYaml(updateValue?: (value: string) => void) {
    const { loading: defaultYamlLoading } = useRequest(async () => {
        if (!updateValue) return;

        const [error, resp] = await awaitWrap(modelAPI.getDefaultParamsYaml());
        if (error || !isRequestSuccess(resp)) {
            return;
        }

        const data = getResponseData(resp);
        const result = data?.params_yaml || '';
        updateValue?.(result);

        return result;
    });

    return {
        defaultYamlLoading,
    };
}
