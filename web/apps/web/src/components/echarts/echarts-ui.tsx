import { forwardRef } from 'react';

export interface EchartsUIProps {
    width?: string;
    height?: string;
}

/**
 * echarts ui components
 */
const EchartsUI = forwardRef<HTMLDivElement, EchartsUIProps>((props, ref) => {
    const { width = '100%', height = '300px' } = props;

    return (
        <div
            ref={ref}
            style={{
                width,
                height,
            }}
        />
    );
});

export default EchartsUI;
