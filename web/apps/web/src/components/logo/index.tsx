import React from 'react';

import CamThinkImg from '@/assets/cam-think.svg';

import './style.less';

const Logo: React.FC = () => {
    return (
        <div className="ms-logo" onClick={() => window?.location?.reload?.()}>
            <img src={CamThinkImg} role="img" alt="" />
        </div>
    );
};

export default Logo;
