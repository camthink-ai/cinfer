import React, { forwardRef, useImperativeHandle, useRef, useMemo } from 'react';
import { useControllableValue } from 'ahooks';
import { type ReactCodeMirrorRef } from '@uiw/react-codemirror';
import classNames from 'classnames';
import { EditorHeaderComponent, EditorComponent } from './components';
import { useCssVariable, useEditorCommand, useEditorTheme } from './hooks';
import type { EditorSupportLang, EditorProps, EditorHandlers } from './types';
import './style.less';

export const CodeEditor = forwardRef<EditorHandlers, EditorProps>((props, ref) => {
    const {
        title,
        icon,
        readOnly = false,
        editable = true,
        height = '100%',
        renderHeader,
        onBlur,
        onFocus,
        supportLangs,
        theme,
        fontSize,
        error,
        ...rest
    } = props;
    const editorRef = useRef<HTMLDivElement>(null);
    const editorInstanceRef = useRef<ReactCodeMirrorRef>(null);
    const { handleBlur, handleFocus, themeBgColor } = useCssVariable({
        onBlur,
        onFocus,
        editorRef,
    });
    const { editorTheme } = useEditorTheme({ fontSize });

    const [editorLang, setEditorLang] = useControllableValue<EditorSupportLang>(props, {
        defaultValuePropName: 'defaultEditorLang',
        valuePropName: 'editorLang',
        trigger: 'onLangChange',
    });
    const [editorValue, setEditorValue] = useControllableValue<string>(props, {
        defaultValuePropName: 'defaultValue',
        valuePropName: 'value',
        trigger: 'onChange',
    });

    const { handlers } = useEditorCommand({ editorInstanceRef, readOnly, editable });
    /** Methods exposed to external components */
    useImperativeHandle(ref, () => handlers);

    const codeEditorCls = useMemo(() => {
        return classNames('ms-code-editor', {
            error: Boolean(error),
        });
    }, [error]);

    return (
        <div className={codeEditorCls} ref={editorRef}>
            <EditorHeaderComponent
                title={title}
                icon={icon}
                editorHandlers={handlers}
                editorLang={editorLang}
                editorValue={editorValue}
                setEditorLang={setEditorLang}
                readOnly={readOnly}
                editable={editable}
                renderHeader={renderHeader}
                style={themeBgColor}
                supportLangs={supportLangs}
            />
            <EditorComponent
                {...rest}
                theme={theme || editorTheme}
                ref={editorInstanceRef}
                editorLang={editorLang}
                editorValue={editorValue}
                setEditorValue={setEditorValue}
                readOnly={readOnly}
                editable={editable}
                height={height}
                onFocus={handleFocus}
                onBlur={handleBlur}
            />
        </div>
    );
});

export default React.memo(CodeEditor);
