import React from 'react';
import { Grid2 as Grid } from '@mui/material';
import { BasicInfo } from './components';

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
                    <Grid size={2}>
                        <BasicInfo />
                    </Grid>
                    <Grid
                        container
                        size={10}
                        sx={{
                            background: '#81DBCF',
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
