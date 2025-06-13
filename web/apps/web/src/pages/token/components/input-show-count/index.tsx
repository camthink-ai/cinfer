import React from 'react';

import './style.less';

export interface InputShowCountProps {
    count: number;
    maxLength: number;
    children: React.ReactNode;
}

const InputShowCount: React.FC<InputShowCountProps> = props => {
    const { count, maxLength, children } = props;

    return (
        <div className="input-show-count">
            {children}
            <div className="count-max">{`${count}/${maxLength}`}</div>
        </div>
    );
};

export default InputShowCount;
