import React, { useRef, useEffect } from 'react';

import { EchartsUI } from '@/components';
import { useEcharts } from '@/components/echarts';

/**
 * cup usage chart
 */
const CPUUsage: React.FC = () => {
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
                text: '24%',
                textAlign: 'left',
            },
            series: [
                {
                    data: [11, 20, 60, 10, 33, 55, 64, 18, 36, 70, 42, 23, 13, 80, 40, 12, 22],
                    itemStyle: {
                        color: '#FF7C42',
                    },
                    smooth: true,
                    type: 'line',
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
            },
            xAxis: {
                axisTick: {
                    show: false,
                },
                boundaryGap: false,
                data: Array.from({ length: 24 }).map((_item, index) => `${index}:00`),
                splitLine: {
                    lineStyle: {
                        type: 'solid',
                        width: 1,
                    },
                    show: true,
                },
                type: 'category',
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
                    fillerColor: '#ffead98a',
                },
            ],
        });
    }, []);

    return <EchartsUI ref={cpuChartRef} height="280px" />;
};

export default CPUUsage;
