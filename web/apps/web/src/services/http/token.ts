import { client, attachAPI, API_PREFIX } from './client';

export interface TokenItemProps {
    id: ApiKey;
    name: string;
    token: string;
    allowed_models: string[];
    ip_whitelist: string[];
    rate_limit: number;
    monthly_limit: number;
    remaining_requests: number;
    remark: string;
    status: string;
    created_at: number;
    updated_at: number;
}

export interface TokenAPISchema extends APISchema {
    /** add new token */
    addNewToken: {
        request: {
            name: string;
            allowed_models: ApiKey[];
            rate_limit: number;
            monthly_limit?: number;
            ip_whitelist?: string[];
            remark?: string;
        };
        response: TokenItemProps;
    };
    /** get tokens list */
    getTokensList: {
        request: SearchRequestType & {
            status?: string;
        };
        response: TokenItemProps[];
    };
    /** get token detail */
    getTokenDetail: {
        request: {
            access_token_id: ApiKey;
        };
        response: TokenItemProps;
    };
    /** update token */
    updateToken: {
        request: {
            access_token_id: ApiKey;
            name: string;
            allowed_models: ApiKey[];
            rate_limit: number;
            ip_whitelist?: string[];
            monthly_limit?: number;
            remark?: string;
        };
        response: void;
    };
    /** delete token */
    deleteToken: {
        request: {
            access_token_id: ApiKey;
        };
        response: void;
    };
    /** disable token */
    disableToken: {
        request: {
            access_token_id: ApiKey;
        };
        response: void;
    };
    /** enable token */
    enableToken: {
        request: {
            access_token_id: ApiKey;
        };
        response: void;
    };
}

/**
 * Dashboard API services
 */
export default attachAPI<TokenAPISchema>(client, {
    apis: {
        addNewToken: `POST ${API_PREFIX}/tokens`,
        getTokensList: `GET ${API_PREFIX}/tokens`,
        getTokenDetail: `GET ${API_PREFIX}/tokens/:access_token_id`,
        updateToken: `PUT ${API_PREFIX}/tokens/:access_token_id`,
        deleteToken: `DELETE ${API_PREFIX}/tokens/:access_token_id`,
        disableToken: `POST ${API_PREFIX}/tokens/:access_token_id/disable`,
        enableToken: `POST ${API_PREFIX}/tokens/:access_token_id/enable`,
    },
});
