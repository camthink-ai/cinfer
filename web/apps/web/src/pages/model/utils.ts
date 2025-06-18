import { type ModelAPISchema } from '@/services/http';
import { type OperateModelProps } from './components/operate-model-modal';

/**
 * convert Add Model data to conform to the back-end data structure
 */
export const convertAddModelData = (
    data: OperateModelProps,
): ModelAPISchema['addModel']['request'] | null => {
    const { name, engineType, modelFile, paramsYaml, remark } = data || {};
    if (!name || !engineType || !modelFile?.original || !paramsYaml) {
        return null;
    }

    return {
        name,
        engine_type: engineType,
        model_file: modelFile.original,
        params_yaml: paramsYaml,
        remark: remark || undefined,
    };
};
