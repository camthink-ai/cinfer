import { useMemo } from 'react';
import { Stack, Skeleton, SkeletonOwnProps } from '@mui/material';

import './style.less';

type CusSkeletonType = {
    variant?: SkeletonOwnProps['variant'];
    animation?: SkeletonOwnProps['animation'];
};

/** custom skeleton */
function CusSkeleton(props: CusSkeletonType) {
    const { variant = 'rectangular', animation = 'wave' } = props;
    return <Skeleton variant={variant} animation={animation} />;
}

/** layout Skeleton */
function LayoutSkeleton() {
    // top element
    const topSkeleton = useMemo(() => {
        const sideTop = Array.from({ length: 7 });
        return (
            <div>
                {sideTop.map((value: unknown, index: number) => (
                    // eslint-disable-next-line react/no-array-index-key
                    <CusSkeleton key={index} />
                ))}
                <CusSkeleton />
            </div>
        );
    }, []);

    return (
        <Stack direction="row" sx={{ flex: 1 }}>
            <Stack
                sx={{
                    flex: 1,
                    borderLeft: '1px solid var(--border-color-base)',
                }}
            >
                <div className="ms-skeleton-top">{topSkeleton}</div>
                <Skeleton
                    variant="rectangular"
                    animation="wave"
                    sx={{ flex: 1, backgroundColor: 'var(--gray-2)' }}
                    style={{ marginTop: 0 }}
                />
            </Stack>
        </Stack>
    );
}

export default LayoutSkeleton;
