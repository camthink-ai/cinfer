import { useMemo } from 'react';
import { Stack, IconButton } from '@mui/material';
import { useI18n, useTime } from '@milesight/shared/src/hooks';
import {
    DeleteOutlineIcon,
    EditIcon,
    SendOutlinedIcon,
    ArrowCircleDownOutlinedIcon,
    FilterAltIcon,
} from '@milesight/shared/src/components';
import { Tooltip, type ColumnType, PublicationStatus } from '@/components';
import { type ModelItemProps } from '@/services/http';
import { useGlobalStore } from '@/stores';

type OperationType = 'edit' | 'delete' | 'enable';

export type TableRowDataType = ModelItemProps;

export interface UseColumnsProps<T> {
    /**
     * operation Button callbacks
     */
    onButtonClick: (type: OperationType, record: T, otherProp?: unknown) => void;
    /**
     * filtered info
     */
    filteredInfo: Record<string, any>;
}

const useColumns = <T extends TableRowDataType>({
    onButtonClick,
    filteredInfo,
}: UseColumnsProps<T>) => {
    const { getIntlText } = useI18n();
    const { getTimeFormat } = useTime();
    const { inferEngines } = useGlobalStore();

    const columns: ColumnType<T>[] = useMemo(() => {
        return [
            {
                field: 'id',
                headerName: getIntlText('model.title.model_id'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
            },
            {
                field: 'name',
                headerName: getIntlText('model.title.model_name'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
            },
            {
                field: 'status',
                headerName: getIntlText('common.label.status'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
                renderCell({ row }) {
                    return <PublicationStatus hasPublished={row?.status === 'published'} />;
                },
            },
            {
                field: 'engine_type',
                headerName: getIntlText('model.title.infer_engine_type'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
                filteredValue: filteredInfo?.engine_type,
                filterIcon: (filtered: boolean) => {
                    return (
                        <FilterAltIcon
                            sx={{
                                color: filtered ? 'var(--primary-color-7)' : 'var(--gray-color-5)',
                            }}
                        />
                    );
                },
                filters: (inferEngines || []).map(engine => ({
                    text: engine,
                    value: engine,
                })),
            },
            {
                field: 'created_at',
                headerName: getIntlText('common.label.create_time'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
                sortable: true,
                renderCell({ value }) {
                    return getTimeFormat(Number(value));
                },
            },
            {
                field: 'updated_at',
                headerName: getIntlText('common.label.update_time'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
                sortable: true,
                renderCell({ value }) {
                    return getTimeFormat(Number(value));
                },
            },
            {
                field: 'remark',
                headerName: getIntlText('common.label.remark'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
            },
            {
                field: '$operation',
                headerName: getIntlText('common.label.operation'),
                display: 'flex',
                width: 120,
                align: 'left',
                headerAlign: 'left',
                renderCell({ row }) {
                    const isPublished = row?.status === 'published';
                    const statusTitle = isPublished
                        ? getIntlText('common.label.unpublish')
                        : getIntlText('common.label.publish');

                    const isDisableIcon = isPublished || Boolean(row?.is_built_in);
                    const editIcon = (
                        <IconButton
                            disabled={isDisableIcon}
                            sx={{ width: 30, height: 30 }}
                            onClick={() => onButtonClick('edit', row)}
                        >
                            <EditIcon sx={{ width: 20, height: 20 }} />
                        </IconButton>
                    );

                    const deleteIcon = (
                        <IconButton
                            disabled={isDisableIcon}
                            sx={{
                                width: 30,
                                height: 30,
                                color: 'text.secondary',
                            }}
                            onClick={() => onButtonClick('delete', row)}
                        >
                            <DeleteOutlineIcon sx={{ width: 20, height: 20 }} />
                        </IconButton>
                    );

                    return (
                        <Stack
                            direction="row"
                            spacing="4px"
                            sx={{ height: '100%', alignItems: 'center', justifyContent: 'end' }}
                        >
                            {isDisableIcon ? (
                                editIcon
                            ) : (
                                <Tooltip title={getIntlText('common.button.edit')}>
                                    {editIcon}
                                </Tooltip>
                            )}
                            <Tooltip title={statusTitle}>
                                <IconButton
                                    sx={{ width: 30, height: 30 }}
                                    onClick={() => onButtonClick('enable', row, !isPublished)}
                                >
                                    {isPublished ? (
                                        <ArrowCircleDownOutlinedIcon
                                            sx={{ width: 20, height: 20 }}
                                        />
                                    ) : (
                                        <SendOutlinedIcon sx={{ width: 20, height: 20 }} />
                                    )}
                                </IconButton>
                            </Tooltip>
                            {isDisableIcon ? (
                                deleteIcon
                            ) : (
                                <Tooltip title={getIntlText('common.label.delete')}>
                                    {deleteIcon}
                                </Tooltip>
                            )}
                        </Stack>
                    );
                },
            },
        ];
    }, [getIntlText, onButtonClick, getTimeFormat, inferEngines, filteredInfo]);

    return columns;
};

export default useColumns;
