import React from 'react';

import CamThinkImg from '@/assets/cam-think.png';

import './style.less';

const Logo: React.FC = () => {
    return (
        <div className="ms-logo" onClick={() => window?.location?.reload?.()}>
            <img src={CamThinkImg} alt="" />
        </div>
    );
};

export default Logo;
