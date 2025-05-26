import React, { useMemo } from 'react';

import { useI18n, type LangType } from '@milesight/shared/src/hooks';
import { Select } from '@milesight/shared/src/components';
import { Logo } from '@/components';

import './style.less';

/**
 * auth page header
 */
const AuthHeader: React.FC = () => {
    const { lang, langs, changeLang, getIntlText } = useI18n();

    /**
     * language select options
     */
    const options = useMemo(() => {
        return Object.values(langs).map(item => {
            return {
                label: getIntlText(item.labelIntlKey),
                value: item.key,
            };
        });
    }, [langs, getIntlText]);

    return (
        <div className="auth-header">
            <Logo />
            <div className="auth-header__language">
                <Select
                    value={lang}
                    size="small"
                    options={options}
                    onChange={e => {
                        const newLang = e?.target?.value as LangType;
                        if (!newLang) return;

                        changeLang(newLang);
                    }}
                />
            </div>
        </div>
    );
};

export default AuthHeader;
