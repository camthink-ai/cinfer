import { isEmpty } from 'lodash-es';

import { type TokenAPISchema, type TokenItemProps } from '@/services/http';
import type { OperateTokenProps } from './components/operate-token-modal';

/**
 * split ip whitelist by , or ;
 */
const splitIpWhiteList = (ipWhitelist?: string) => {
    if (!ipWhitelist || typeof ipWhitelist !== 'string') return undefined;

    const splitOne = ipWhitelist.split(',');
    if (!Array.isArray(splitOne) || isEmpty(splitOne)) {
        return [ipWhitelist];
    }

    return splitOne
        .reduce((acc: string[], cur) => {
            return [...acc, ...cur.split(';')];
        }, [])
        .filter(Boolean);
};

/**
 * convert data to conform to the back-end data structure
 */
export const convertData = (data: OperateTokenProps): TokenAPISchema['addNewToken']['request'] => {
    const { name, allowedModels, rateLimit, monthlyLimit, ipWhitelist, remark } = data || {};

    return {
        name,
        allowed_models: allowedModels,
        rate_limit: Number(rateLimit),
        monthly_limit: monthlyLimit ? Number(monthlyLimit) : undefined,
        ip_whitelist: splitIpWhiteList(ipWhitelist),
        remark: remark || undefined,
    };
};

/**
 * convert data to conform to the front-end data structure
 */
export const convertDataToDisplay = (data: TokenItemProps): OperateTokenProps => {
    return {
        name: data.name,
        allowedModels: data.allowed_models,
        rateLimit: String(data.rate_limit),
        monthlyLimit: data?.monthly_limit ? String(data.monthly_limit) : undefined,
        ipWhitelist: (data?.ip_whitelist || []).join(';'),
        remark: data.remark,
    };
};
