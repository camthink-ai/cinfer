import { client, unauthClient, attachAPI, API_PREFIX } from './client';

export interface GlobalAPISchema extends APISchema {
    /** Log in */
    oauthLogin: {
        request: {
            /** username */
            username: string;
            /** cipher */
            password: string;
            /** Authorization type */
            grant_type: 'password';
            /** Client ID  */
            client_id: string;
            /** Client Secret  */
            client_secret: string;
        };
        response: {
            /** Authentication Token */
            access_token: string;
            /** Refresh Token */
            refresh_token: string;
            /** Expiration time, unit s */
            // expires_in: number;
        };
    };

    /** User registration */
    oauthRegister: {
        request: {
            email: string;
            nickname: string;
            password: string;
        };
        response: GlobalAPISchema['oauthLogin']['response'];
    };

    /** Get user registration status */
    getUserStatus: {
        request: void;
        response: {
            init: boolean;
        };
    };

    /** Get upload configuration */
    getUploadConfig: {
        request: {
            name?: string;
            file_name: string;
            description?: string;
        };
        response: {
            key: string;
            upload_url: string;
            resource_url: string;
        };
    };

    /** Upload file */
    fileUpload: {
        request: {
            url: string;
            file: File;
            mimeType: string;
        };
        response: unknown;
    };
}

/**
 * Global API services (including registration, login, users, etc.)
 */
export default attachAPI<GlobalAPISchema>(client, {
    apis: {
        oauthLogin: {
            method: 'POST',
            path: `${API_PREFIX}/oauth2/token`,
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
        },
        oauthRegister: `POST ${API_PREFIX}/user/register`,
        getUserStatus: `GET ${API_PREFIX}/user/status`,
        getUserInfo: `GET ${API_PREFIX}/user`,
        getUploadConfig: `POST ${API_PREFIX}/resource/upload-config`,
        async fileUpload(params, options) {
            const { url, file, mimeType } = params;
            const apiUrl = url.startsWith('http')
                ? url
                : `${API_PREFIX}${url.startsWith('/') ? '' : '/'}${url}`;

            return unauthClient.request({
                method: 'PUT',
                url: apiUrl,
                headers: {
                    'Content-Type': mimeType,
                },
                data: file,
                ...options,
            });
        },
    },
});
