import React, { useMemo } from 'react';
import { Divider } from '@mui/material';
import { useNavigate } from 'react-router';
import { useMemoizedFn } from 'ahooks';

import { useI18n } from '@milesight/shared/src/hooks';
import { type DashboardAPISchema } from '@/services/http';

import styles from './style.module.less';

type ItemType = { name: string; value: string | number; path?: string };

export interface BasicInfoProps {
    data?: DashboardAPISchema['getSystemInfo']['response'];
}

/**
 * System Basic Info
 */
const BasicInfo: React.FC<BasicInfoProps> = props => {
    const { data } = props;

    const navigate = useNavigate();
    const { getIntlText } = useI18n();

    /**
     * basic info items
     */
    const infoItems: ItemType[] = useMemo(() => {
        return [
            {
                name: getIntlText('dashboard.label.system_version'),
                value: `${data?.os_info?.os_name || ''} ${data?.os_info?.os_version || ''}`,
            },
            {
                name: getIntlText('dashboard.label.software_name'),
                value: data?.software_name || '',
            },
            {
                name: getIntlText('dashboard.label.hardware_speed'),
                value: (data?.hardware_acceleration || []).join('/'),
            },
            {
                name: getIntlText('dashboard.label.version'),
                value: data?.software_version || '',
            },
        ];
    }, [getIntlText, data]);

    /**
     * statistics data items
     */
    const statisticsItems: ItemType[] = useMemo(() => {
        return [
            {
                name: getIntlText('common.label.model'),
                value: data?.models_stats?.total_count || 0,
                path: '/model',
            },
            {
                name: 'Token',
                value: data?.access_tokens_stats?.total_count || 0,
                path: '/token',
            },
        ];
    }, [getIntlText, data]);

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
            <div className={styles.header}>{getIntlText('dashboard.title.host_info')}</div>

            {infoItems.map(item => (
                <div key={item.name} className={styles.item}>
                    <div className={styles.item__name}>{item.name}</div>
                    <div className={styles.item_content}>{item.value}</div>
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
