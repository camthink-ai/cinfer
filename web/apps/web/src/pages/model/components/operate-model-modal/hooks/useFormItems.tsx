import { useMemo, useRef } from 'react';
import { type ControllerProps } from 'react-hook-form';
import { TextField, FormControl, InputLabel, FormHelperText } from '@mui/material';

import { useI18n } from '@milesight/shared/src/hooks';
import { Select, OpenInFullIcon, CloseFullscreenIcon } from '@milesight/shared/src/components';
import { checkRequired, isMaxBytesLength } from '@milesight/shared/src/utils/validators';

import { InputShowCount, Upload, type FileValueType, CodeEditor, Tooltip } from '@/components';
import { useGlobalStore } from '@/stores';
import { type OperateModelProps } from '../index';

import styles from '../style.module.less';

export function useFormItems(props: {
    yamlFullscreen: boolean;
    toggleYamlFullscreen: (isFullScreen: boolean) => void;
    engineType: string;
}) {
    const { yamlFullscreen, toggleYamlFullscreen, engineType } = props;

    const { getIntlText } = useI18n();
    const { inferEngines } = useGlobalStore();

    const engineOptions = useMemo(() => {
        return (inferEngines || []).map(engine => ({ label: engine, value: engine }));
    }, [inferEngines]);

    const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

    const formItems = useMemo((): ControllerProps<OperateModelProps>[] => {
        return [
            {
                name: 'name',
                rules: {
                    maxLength: {
                        value: 300,
                        message: getIntlText('valid.input.max_length', {
                            1: 300,
                        }),
                    },
                    validate: {
                        checkRequired: checkRequired(),
                    },
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            required
                            fullWidth
                            type="text"
                            label={getIntlText('common.label.name')}
                            placeholder={getIntlText('common.placeholder.input')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                        />
                    );
                },
            },
            {
                name: 'engineType',
                rules: {
                    validate: {
                        checkRequired: checkRequired(),
                    },
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <Select
                            fullWidth
                            required
                            placeholder={getIntlText('common.placeholder.select')}
                            label={getIntlText('model.title.infer_engine_type')}
                            options={engineOptions}
                            value={value as ApiKey}
                            onChange={onChange}
                            error={error}
                        />
                    );
                },
            },
            {
                name: 'modelFile',
                rules: {
                    validate: {
                        checkRequired: checkRequired(),
                        checkFileExt: file => {
                            const fileName = (file as FileValueType)?.name;
                            if (!fileName || !engineType) {
                                return true;
                            }

                            const ext = fileName.slice(fileName.lastIndexOf('.') + 1);
                            if (ext === engineType) {
                                return true;
                            }

                            return getIntlText('common.message.upload_error_file_invalid_type', {
                                1: `.${engineType}`,
                            });
                        },
                    },
                },
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <FormControl fullWidth>
                            <Upload
                                required
                                value={value as FileValueType}
                                onChange={onChange}
                                label={getIntlText('model.title.model_file')}
                                multiple={false}
                                accept={{
                                    '*': engineType ? [`.${engineType}`] : [],
                                }}
                                error={error}
                                maxSize={1000 * 1024 * 1024}
                                autoUpload={false}
                                ignoreMimeTypeWarn
                            />
                        </FormControl>
                    );
                },
            },
            {
                name: 'paramsYaml',
                rules: {
                    validate: {
                        checkRequired: checkRequired(),
                        checkMaxBytes: str => {
                            return isMaxBytesLength(String(str || ''), 1 * 1024 * 1024)
                                ? true
                                : getIntlText('valid.input.length', {
                                      0: '1MB',
                                  });
                        },
                    },
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    const yamlEditor = (
                        <CodeEditor
                            editorLang="yaml"
                            title="YAML"
                            value={value as string}
                            onChange={onChange}
                            error={!!error}
                            icon={
                                yamlFullscreen ? (
                                    <Tooltip
                                        title={getIntlText('common.label.exit_fullscreen_editor')}
                                    >
                                        <CloseFullscreenIcon
                                            className="ms-header-copy"
                                            onClick={() => toggleYamlFullscreen(false)}
                                        />
                                    </Tooltip>
                                ) : (
                                    <Tooltip title={getIntlText('common.label.fullscreen_editor')}>
                                        <OpenInFullIcon
                                            className="ms-header-copy"
                                            onClick={() => toggleYamlFullscreen(true)}
                                        />
                                    </Tooltip>
                                )
                            }
                        />
                    );

                    return yamlFullscreen ? (
                        <div className={styles['yaml-editor-fullscreen']}>{yamlEditor}</div>
                    ) : (
                        <FormControl
                            fullWidth
                            sx={{
                                height: '188px',
                            }}
                        >
                            <InputLabel required>
                                {getIntlText('model.title.model_parameter')}
                            </InputLabel>
                            {yamlEditor}
                            <FormHelperText error={!!error}>{error?.message || ''}</FormHelperText>
                        </FormControl>
                    );
                },
            },
            {
                name: 'remark',
                rules: {
                    maxLength: {
                        value: 1000,
                        message: getIntlText('valid.input.max_length', {
                            1: 1000,
                        }),
                    },
                },
                defaultValue: '',
                render({ field: { onChange, value }, fieldState: { error } }) {
                    return (
                        <TextField
                            inputRef={inputRef}
                            fullWidth
                            type="text"
                            multiline
                            rows={3}
                            label={getIntlText('common.label.remark')}
                            placeholder={getIntlText('common.placeholder.input')}
                            error={!!error}
                            helperText={error ? error.message : null}
                            value={value}
                            onChange={onChange}
                            onBlur={event => {
                                const newValue = event?.target?.value;
                                onChange(typeof newValue === 'string' ? newValue.trim() : newValue);
                            }}
                            sx={{
                                '.MuiInputBase-root': {
                                    padding: '0 12px 16px',
                                },
                            }}
                            slotProps={{
                                input: {
                                    endAdornment: (
                                        <InputShowCount
                                            inputRef={inputRef}
                                            value={value}
                                            maxLength={1000}
                                        />
                                    ),
                                },
                            }}
                        />
                    );
                },
            },
        ];
    }, [getIntlText, engineOptions, yamlFullscreen, toggleYamlFullscreen, engineType]);

    return {
        formItems,
    };
}
