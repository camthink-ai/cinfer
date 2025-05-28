import React, { useMemo } from 'react';
import { Divider } from '@mui/material';
import { useNavigate } from 'react-router';
import { useMemoizedFn } from 'ahooks';

import { useI18n } from '@milesight/shared/src/hooks';

import styles from './style.module.less';

type ItemType = { name: string; value: string | number; path?: string };

/**
 * System Basic Info
 */
const BasicInfo: React.FC = () => {
    const navigate = useNavigate();
    const { getIntlText } = useI18n();

    /**
     * basic info items
     */
    const infoItems: ItemType[] = useMemo(() => {
        return [
            {
                name: getIntlText('dashboard.label.system_version'),
                value: 'ubantu22.04',
            },
            {
                name: getIntlText('dashboard.label.software_name'),
                value: 'CamThink AI Inference Platform',
            },
            {
                name: getIntlText('dashboard.label.hardware_speed'),
                value: 'CPU/GPU/TensorRT',
            },
            {
                name: getIntlText('dashboard.label.version'),
                value: 'v1.0',
            },
        ];
    }, [getIntlText]);

    /**
     * statistics data items
     */
    const statisticsItems: ItemType[] = useMemo(() => {
        return [
            {
                name: getIntlText('dashboard.label.model'),
                value: 180,
                path: '/model',
            },
            {
                name: 'Token',
                value: 6668,
                path: '/token',
            },
        ];
    }, [getIntlText]);

    /**
     * Jump to page
     */
    const handleNavigate = useMemoizedFn((path?: string) => {
        if (!path) return;

        navigate(path);
    });

    /**
     * if the data is greater than 1000, perform the conversion
     */
    const conversionNumber = useMemoizedFn((num: ItemType['value']) => {
        if (!num) return num;

        const newNum = Number(num);
        if (Number.isNaN(newNum) || newNum < 1000) return num;

        return `${(newNum / 1000).toFixed(1).replace(/\.0$/, '')}k`;
    });

    return (
        <div className={styles['basic-info']}>
            <div className={styles.header}>Neo Edge NG4500</div>

            {infoItems.map(item => (
                <div key={item.name} className={styles.item}>
                    <div className={styles.item__name}>{item.name}</div>
                    <div>{item.value}</div>
                </div>
            ))}

            <div className={styles.divider}>
                <Divider component="div" />
            </div>

            <div className={styles.statistics}>
                {statisticsItems.map(item => (
                    <div key={item.name} className={styles.data}>
                        <div className={styles.data__name}>{item.name}</div>
                        <div
                            className={styles.data__number}
                            onClick={() => handleNavigate(item.path)}
                        >
                            {conversionNumber(item.value)}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default BasicInfo;
