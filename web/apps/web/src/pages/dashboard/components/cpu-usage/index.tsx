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
                    data: [
                        ['2024-05-29 17:23:41', 12],
                        ['2024-05-29 19:47:15', 34],
                        ['2024-05-29 20:12:58', 56],
                        ['2024-05-29 21:39:21', 78],
                        ['2024-05-29 22:05:44', 23],
                        ['2024-05-29 23:54:36', 67],
                        ['2024-05-30 01:16:59', 45],
                        ['2024-05-30 02:03:20', 31],
                        ['2024-05-30 02:57:02', 58],
                        ['2024-05-30 03:45:11', 72],
                        ['2024-05-30 04:19:33', 64],
                        ['2024-05-30 05:22:47', 81],
                        ['2024-05-30 06:11:29', 27],
                        ['2024-05-30 06:55:56', 53],
                        ['2024-05-30 07:37:18', 96],
                        ['2024-05-30 08:25:09', 42],
                        ['2024-05-30 08:59:51', 77],
                        ['2024-05-30 09:14:05', 68],
                        ['2024-05-30 09:48:27', 35],
                        ['2024-05-30 10:31:32', 80],
                        ['2024-05-30 10:59:49', 15],
                        ['2024-05-30 11:21:17', 62],
                        ['2024-05-30 12:07:43', 44],
                        ['2024-05-30 12:53:28', 90],
                        ['2024-05-30 13:36:55', 29],
                        ['2024-05-30 14:20:06', 55],
                        ['2024-05-30 15:02:14', 73],
                        ['2024-05-30 15:15:38', 60],
                        ['2024-05-30 15:29:25', 38],
                    ],
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
                    fillerColor: '#ffead98a',
                    showDetail: false,
                },
            ],
        });
    }, []);

    return <EchartsUI ref={cpuChartRef} height="280px" />;
};

export default CPUUsage;
