import axios from 'axios';
import type { AxiosRequestConfig } from 'axios';
import {
    apiOrigin,
    oauthClientID,
    oauthClientSecret,
    REFRESH_TOKEN_TOPIC,
} from '@milesight/shared/src/config';
import eventEmitter from '@milesight/shared/src/utils/event-emitter';
import { getResponseData } from '@milesight/shared/src/utils/request';
import iotStorage, { TOKEN_CACHE_KEY } from '@milesight/shared/src/utils/storage';
import { API_PREFIX } from './constant';

type TokenDataType = {
    /** Authentication Token */
    access_token: string;
    /** Refresh Token */
    refresh_token: string;
    /**
     * Expiration time, unit: ms
     *
     * Note: This value is the front-end expiration time and is only used to determine when the token needs to be refreshed. The actual token may not have expired at the back-end
     */
    expires_in: number;
};

let timer: number | null = null;
/** Token delay refreshing time */
const REFRESH_TOKEN_TIMEOUT = 1 * 1000;
/** Token refresh API path */
const tokenApiPath = `${API_PREFIX}/auth/refresh-token`;
/**
 * Generate Authorization request header data
 * @param token Login certificate
 */

/**
 * Token Processing Logic (Silent processing)
 *
 * 1. Check whether the token in the cache is valid. If yes, write the token into the request header
 * 2. Refresh the token periodically every 60 minutes
 */
const oauthHandler = async (config: AxiosRequestConfig) => {
    const token = iotStorage.getItem<TokenDataType>(TOKEN_CACHE_KEY);
    const isExpired = token && Date.now() >= token.expires_in;
    const isOauthRequest = ['auth/login', 'auth/refresh-token'].some(path =>
        config.url?.includes(path),
    );

    if (token?.access_token && !isOauthRequest) {
        config.headers = config.headers || {};
        config.headers['X-Auth-Token'] = token?.access_token;
    }

    /**
     * 1. If the request is oauth, the token is not refreshed
     * 2. If there is no local cache token, do not refresh the token
     * 3. If the local cache token does not expire, do not refresh the token
     */
    if (isOauthRequest || !token?.access_token || !isExpired) {
        return config;
    }

    /**
     * After one second of delay, a token update request is sent to ensure that the request using the old token can still pass the back-end authentication within one second
     */
    if (timer) window.clearTimeout(timer);
    timer = window.setTimeout(() => {
        const requestConfig = {
            baseURL: apiOrigin,
            headers: {
                'X-Auth-Token': token?.access_token,
            },
            withCredentials: true,
        };
        const requestData = {
            refresh_token: token.refresh_token,
        };

        axios
            .post<ApiResponse<TokenDataType>>(tokenApiPath, requestData, requestConfig)
            .then(resp => {
                const data = getResponseData(resp)!;

                // The token is refreshed every 60 minutes
                data.expires_in = Date.now() + 60 * 60 * 1000;
                iotStorage.setItem(TOKEN_CACHE_KEY, data);
                eventEmitter.publish(REFRESH_TOKEN_TOPIC);
            })
            .catch(_ => {
                // TODO: If the token is invalid, the token is directly removed
                // iotStorage.removeItem(TOKEN_CACHE_KEY);
            });
    }, REFRESH_TOKEN_TIMEOUT);

    return config;
};

export default oauthHandler;
