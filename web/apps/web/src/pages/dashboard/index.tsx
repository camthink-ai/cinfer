import React from 'react';
import { Grid2 as Grid, Stack } from '@mui/material';

import './style.less';

const Dashboard: React.FC = () => {
    return (
        <div className="ms-main-blank">
            <div className="ms-view-dashboard">
                <Grid
                    container
                    spacing={2}
                    sx={{
                        height: '100%',
                    }}
                >
                    <Grid
                        size={2}
                        sx={{
                            background: 'purple',
                        }}
                    >
                        <Stack spacing={2}>
                            <div>Column 1 - Row 1</div>
                            <div>Column 1 - Row 2</div>
                            <div>Column 1 - Row 3</div>
                        </Stack>
                    </Grid>
                    <Grid
                        container
                        size={10}
                        sx={{
                            background: 'orange',
                        }}
                    >
                        <Grid size={6}>1</Grid>
                        <Grid size={6}>2</Grid>
                        <Grid size={6}>3</Grid>
                        <Grid size={6}>other</Grid>
                    </Grid>
                </Grid>
            </div>
        </div>
    );
};

export default Dashboard;
