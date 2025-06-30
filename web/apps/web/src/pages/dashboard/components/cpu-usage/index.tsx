import React, { useRef, useEffect } from 'react';

import { EchartsUI } from '@/components';
import { useEcharts } from '@/components/echarts';

export interface CPUUsageProps {
    cpuData: (string | number)[][];
}

/**
 * cup usage chart
 */
const CPUUsage: React.FC<CPUUsageProps> = props => {
    const { cpuData } = props;

    const cpuChartRef = useRef<HTMLDivElement>(null);

    const { renderEcharts } = useEcharts(cpuChartRef);

    useEffect(() => {
        renderEcharts({
            grid: {
                containLabel: true,
                top: '42px',
                left: '1%',
            },
            title: {
                text: `${cpuData?.[cpuData.length - 1]?.[1] || 0}%`,
                textAlign: 'left',
            },
            series: [
                {
                    type: 'line',
                    smooth: true,
                    itemStyle: {
                        color: '#FF7C42',
                    },
                    data: cpuData,
                },
            ],
            tooltip: {
                axisPointer: {
                    lineStyle: {
                        color: '#FF7C42',
                        width: 1,
                    },
                },
                trigger: 'axis',
                valueFormatter(value) {
                    return `${value}%`;
                },
            },
            xAxis: {
                type: 'time',
                boundaryGap: [0, 0],
            },
            yAxis: {
                axisTick: {
                    show: false,
                },
                max: 100,
                interval: 25,
                type: 'value',
                boundaryGap: [0, '100%'],
                axisLabel: {
                    formatter(value) {
                        return `${value}%`;
                    },
                },
            },
            dataZoom: [
                {
                    type: 'inside',
                    start: 0,
                    end: 100,
                },
                {
                    type: 'slider',
                    start: 0,
                    end: 100,
                    fillerColor: 'rgba(255, 124, 66, 0.16)',
                    showDetail: false,
                    moveHandleStyle: {
                        color: '#FF7C42',
                        opacity: 0.16,
                    },
                    emphasis: {
                        handleLabel: {
                            show: true,
                        },
                        moveHandleStyle: {
                            color: '#FF7C42',
                            opacity: 1,
                        },
                    },
                    borderColor: '#E5E6EB',
                    dataBackground: {
                        lineStyle: {
                            color: '#FF7C42',
                            opacity: 0.36,
                        },
                        areaStyle: {
                            color: '#FF7C42',
                            opacity: 0.08,
                        },
                    },
                    selectedDataBackground: {
                        lineStyle: {
                            color: '#FF7C42',
                            opacity: 0.8,
                        },
                        areaStyle: {
                            color: '#FF7C42',
                            opacity: 0.2,
                        },
                    },
                    brushStyle: {
                        color: '#FF7C42',
                        opacity: 0.16,
                    },
                },
            ],
        });
    }, [cpuData]);

    return <EchartsUI ref={cpuChartRef} height="280px" />;
};

export default CPUUsage;
