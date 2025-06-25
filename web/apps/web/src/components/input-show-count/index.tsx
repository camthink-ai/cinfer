import React, { useEffect, useState, useRef } from 'react';

import './style.less';

export interface InputShowCountProps {
    value?: any;
    inputRef?: React.RefObject<HTMLInputElement | HTMLTextAreaElement>;
    maxLength: number;
}

const InputShowCount: React.FC<InputShowCountProps> = props => {
    const { value, maxLength, inputRef } = props;

    const [newCount, setNewCount] = useState<number>(0);
    const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

    /**
     * Get Real input/textarea content value length
     */
    useEffect(() => {
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }
        if (!inputRef?.current) return;

        timeoutRef.current = setTimeout(() => {
            setNewCount(inputRef.current?.value?.length || 0);
        }, 100);
    }, [value, inputRef]);

    return (
        <div className="input-show-count">{`${newCount || String(value || '')?.length || 0}/${maxLength}`}</div>
    );
};

export default InputShowCount;
