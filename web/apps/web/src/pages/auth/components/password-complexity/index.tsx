import React, { useMemo } from 'react';

import { useI18n } from '@milesight/shared/src/hooks';
import { CheckCircleIcon, CancelIcon } from '@milesight/shared/src/components';

import styles from './style.module.less';

export interface PasswordComplexityProps {
    password?: string;
}

/**
 * password complexity validator
 */
const PasswordComplexity: React.FC<PasswordComplexityProps> = props => {
    const { password = '' } = props;
    const { getIntlText } = useI18n();

    const validatePoints: { isPass: boolean; text: string }[] = useMemo(() => {
        return [
            {
                isPass: /[A-Z]/.test(password),
                text: getIntlText('valid.input.one_uppercase_letter'),
            },
            {
                isPass: /[a-z]/.test(password),
                text: getIntlText('valid.input.one_lowercase_letter'),
            },
            {
                isPass: /[0-9]/.test(password),
                text: getIntlText('valid.input.one_number'),
            },
            {
                isPass: password.length >= 8,
                text: getIntlText('valid.input.eight_characters'),
            },
        ];
    }, [getIntlText, password]);

    return (
        <div className={styles['password-complexity']}>
            {validatePoints.map(point => (
                <div className={styles.item}>
                    <div className={point.isPass ? styles['check-icon'] : styles['cancel-icon']}>
                        {point.isPass ? <CheckCircleIcon /> : <CancelIcon />}
                    </div>
                    <div className={styles.text}>{point.text}</div>
                </div>
            ))}
        </div>
    );
};

export default PasswordComplexity;
