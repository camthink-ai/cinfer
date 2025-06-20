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

/**
 * convert Edit Model data to conform to the back-end data structure
 */
export const convertEditModelData = (
    data: OperateModelProps,
): Omit<ModelAPISchema['updateModel']['request'], 'model_id'> => {
    const { name, engineType, modelFile, paramsYaml, remark } = data || {};

    const isFile = Object.prototype.toString.call(modelFile?.original) === '[object File]';
    return {
        name,
        engine_type: engineType,
        model_file: isFile ? modelFile?.original : undefined,
        params_yaml: paramsYaml,
        remark: remark || undefined,
    };
};

/**
 * convert data to conform to the front-end data structure
 */
export const convertDataToDisplay = (
    data?: ModelAPISchema['getModelDetail']['response'],
): OperateModelProps => {
    return {
        name: data?.name || '',
        engineType: data?.engine_type || '',
        modelFile: {
            name: data?.model_file_info?.name || '',
            path: data?.file_path || '',
            size: data?.model_file_info?.size_bytes || 0,
        },
        paramsYaml: data?.params_yaml || '',
        remark: data?.remark || '',
    };
};
