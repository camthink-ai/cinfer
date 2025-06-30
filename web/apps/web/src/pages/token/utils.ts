import { isEmpty, difference } from 'lodash-es';
import intl from 'react-intl-universal';

import { type TValidator, isIP } from '@milesight/shared/src/utils/validators';
import { type TokenAPISchema, type TokenItemProps } from '@/services/http';
import type { OperateTokenProps } from './components/operate-token-modal';

export const ALL_MODELS_SIGN = 'ALL';

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
        monthlyLimit: data?.monthly_limit ? String(data.monthly_limit) : '',
        ipWhitelist: (data?.ip_whitelist || []).join(';'),
        remark: data.remark,
    };
};

/**
 * to validate ip whitelist is correct ipv4
 */
export const checkIPWhitelist: TValidator = () => {
    return value => {
        try {
            const ipList = splitIpWhiteList(value);
            let errorIp = '';
            if (
                ipList &&
                ipList.some(ip => {
                    const isIp = isIP(ip, 4);
                    if (!isIp) {
                        errorIp = ip;
                    }

                    return !isIp;
                })
            ) {
                return intl
                    .get('token.tip.ip_whitelist_validate', {
                        1: `${(errorIp || '').slice(0, 38)}${(errorIp?.length || 0) > 38 ? '...' : ''}`,
                    })
                    .d('token.tip.ip_whitelist_validate');
            }
        } catch (e) {
            // do nothing
        }

        return Promise.resolve(true);
    };
};

export const transformAllModels = (newModels: ApiKey[], oldModels: ApiKey[]) => {
    if (!Array.isArray(newModels) || !Array.isArray(oldModels)) {
        return newModels;
    }

    /**
     * Get the current checked model
     */
    const diff = difference(newModels, oldModels);
    const diffValue = diff?.[0];
    if (!diffValue) {
        return newModels;
    }

    /**
     * If all models are currently selected, deselect all other selected models
     */
    if (newModels.includes(ALL_MODELS_SIGN) && diffValue === ALL_MODELS_SIGN) {
        return [ALL_MODELS_SIGN];
    }

    /**
     * If another models is currently selected, cancel all models selections
     */
    if (
        diffValue !== ALL_MODELS_SIGN &&
        oldModels?.length === 1 &&
        oldModels?.[0] === ALL_MODELS_SIGN
    ) {
        return newModels.filter(m => m !== ALL_MODELS_SIGN);
    }

    return newModels;
};
