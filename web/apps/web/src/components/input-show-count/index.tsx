import React from 'react';

import './style.less';

export interface InputShowCountProps {
    count: number;
    maxLength: number;
}

const InputShowCount: React.FC<InputShowCountProps> = props => {
    const { count, maxLength } = props;

    return <div className="input-show-count">{`${count}/${maxLength}`}</div>;
};

export default InputShowCount;
