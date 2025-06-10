import React, { useRef, useEffect } from 'react';

import { useI18n } from '@milesight/shared/src/hooks';

import { type DashboardAPISchema } from '@/services/http';
import { EchartsUI } from '@/components';
import { useEcharts } from '@/components/echarts';

export interface ModelStatusProps {
    status?: DashboardAPISchema['getSystemInfo']['response']['models_stats'];
}

/**
 * model status chart
 */
const ModelStatus: React.FC<ModelStatusProps> = props => {
    const { status } = props;
    const { getIntlText } = useI18n();

    const modelChartRef = useRef<HTMLDivElement>(null);
    const { renderEcharts } = useEcharts(modelChartRef);

    useEffect(() => {
        /**
         * models status data
         */
        const modelsStatus = () => {
            if (!status?.published_count && !status?.unpublished_count) {
                return [];
            }

            return [
                {
                    name: getIntlText('common.label.already_publish_status'),
                    value: status?.published_count || 0,
                },
                {
                    name: getIntlText('common.label.unpublish_status'),
                    value: status?.unpublished_count || 0,
                    emphasis: { itemStyle: { color: '#F2F3F5' } },
                },
            ];
        };

        renderEcharts({
            legend: {
                bottom: '2%',
                left: 'center',
            },
            series: [
                {
                    animationDelay() {
                        return Math.random() * 100;
                    },
                    animationEasing: 'exponentialInOut',
                    animationType: 'scale',
                    avoidLabelOverlap: false,
                    color: ['#FF7C42', '#F2F3F5'],
                    data: modelsStatus(),
                    emphasis: {
                        label: {
                            fontSize: '12',
                            fontWeight: 'bold',
                            show: true,
                        },
                    },
                    label: {
                        position: 'center',
                        show: false,
                    },
                    labelLine: {
                        show: false,
                    },
                    name: getIntlText('dashboard.label.model_operating_status'),
                    radius: ['40%', '65%'],
                    type: 'pie',
                },
            ],
            tooltip: {
                trigger: 'item',
            },
        });
    }, [status]);

    return <EchartsUI ref={modelChartRef} height="280px" />;
};

export default ModelStatus;
