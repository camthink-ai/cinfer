import { client, attachAPI, API_PREFIX } from './client';

export interface GlobalAPISchema extends APISchema {
    /** Log in */
    oauthLogin: {
        request: {
            /** username */
            username: string;
            /** cipher */
            password: string;
        };
        response: {
            /** Authentication Token */
            access_token: string;
            /** Refresh Token */
            refresh_token: string;
            /** Expiration time, unit s */
            expires_in: number;
        };
    };

    /** User registration */
    oauthRegister: {
        request: {
            username: string;
            password: string;
        };
        response: void;
    };

    /** Get user registration status */
    getUserStatus: {
        request: void;
        response: {
            init: boolean;
        };
    };

    /** Get User Info */
    getUserInfo: {
        request: void;
        response: {
            user_id: ApiKey;
            username: string;
        };
    };

    /**
     * oauth refresh token
     */
    oauthRefreshToken: {
        request: {
            refresh_token: string;
        };
        response: GlobalAPISchema['oauthLogin']['response'];
    };

    /**
     * user logout
     */
    userLogout: {
        request: {
            refresh_token: string;
        };
        response: void;
    };
}

/**
 * Global API services (including registration, login, users, etc.)
 */
export default attachAPI<GlobalAPISchema>(client, {
    apis: {
        oauthLogin: `POST ${API_PREFIX}/auth/login`,
        oauthRegister: `POST ${API_PREFIX}/auth/register`,
        getUserStatus: `GET ${API_PREFIX}/system/status`,
        getUserInfo: `GET ${API_PREFIX}/auth/userInfo`,
        oauthRefreshToken: `POST ${API_PREFIX}/auth/refresh-token`,
        userLogout: `POST ${API_PREFIX}/auth/logout`,
    },
});
