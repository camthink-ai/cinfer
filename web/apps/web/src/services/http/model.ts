import { client, attachAPI, API_PREFIX } from './client';

export type ModelStatusType = 'draft' | 'published' | 'deprecated';

export interface ModelItemProps {
    id: ApiKey;
    name: string;
    remark: string;
    engine_type: string;
    status: ModelStatusType;
    created_at: number;
    updated_at: number;
}

export interface ModelAPISchema extends APISchema {
    /** Models List Data */
    getModelList: {
        request: SearchRequestType & {
            name?: string;
            status?: ModelStatusType;
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
            model_id: ApiKey;
            name?: string;
            model_file?: File;
            params_yaml?: string;
            engine_type?: string;
            remark?: string;
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
    /** get engines data */
    getEngines: {
        request: void;
        response: string[];
    };
    /** get model detail */
    getModelDetail: {
        request: {
            model_id: ApiKey;
        };
        response: {
            id: ApiKey;
            name: string;
            remark?: string;
            engine_type: string;
            input_schema: Record<string, any>[];
            output_schema: Record<string, any>[];
            config: Record<string, any>[];
            file_path: string;
            params_path: string;
            created_by: string;
            created_at: number;
            updated_at: number;
            status: ModelStatusType;
            model_file_info: {
                name: string;
                size_bytes: number;
            };
            params_yaml: string;
        };
    };
}

/**
 * Dashboard API services
 */
export default attachAPI<ModelAPISchema>(client, {
    apis: {
        getModelList: `GET ${API_PREFIX}/models`,
        addModel: {
            method: 'POST',
            path: `POST ${API_PREFIX}/models`,
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        },
        updateModel: {
            method: 'PUT',
            path: `PUT ${API_PREFIX}/models/:model_id`,
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        },
        deleteModel: `DELETE ${API_PREFIX}/models/:model_id`,
        getEngines: `GET ${API_PREFIX}/system/engines`,
        getModelDetail: `GET ${API_PREFIX}/models/:model_id`,
    },
});
