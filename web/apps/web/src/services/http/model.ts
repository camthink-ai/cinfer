import { client, attachAPI, API_PREFIX } from './client';

export type ModelStatusType = 'draft' | 'published' | 'deprecated';

export interface ModelItemProps {
    id: ApiKey;
    name: string;
    remark: string;
    engine_type: string;
    status: ModelItemProps;
    created_at: number;
    updated_at: number;
}

export interface ModelAPISchema extends APISchema {
    /** Models List Data */
    getModelList: {
        request: SearchRequestType & {
            name?: string;
            status?: 'draft' | 'published' | 'deprecated';
        };
        response: ModelItemProps[];
    };
    /** add new model */
    addModel: {
        request: {
            name: string;
            model_file: File;
            params_yaml: string;
            engine_type: string;
            remark?: string;
        };
        response: ModelItemProps;
    };
    /** update model */
    updateModel: {
        request: {
            model_id: string;
        };
        response: ModelItemProps;
    };
    /** delete model */
    deleteModel: {
        request: {
            model_id: string;
        };
        response: void;
    };
}

/**
 * Dashboard API services
 */
export default attachAPI<ModelAPISchema>(client, {
    apis: {
        getModelList: `GET ${API_PREFIX}/models`,
        addModel: `POST ${API_PREFIX}/models`,
        updateModel: `PUT ${API_PREFIX}/models/:model_id`,
        deleteModel: `DELETE ${API_PREFIX}/models/:model_id`,
    },
});
