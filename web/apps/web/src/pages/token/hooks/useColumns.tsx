import { useMemo } from 'react';
import { Stack, IconButton, Switch } from '@mui/material';
import { useI18n, useTime } from '@milesight/shared/src/hooks';
import { DeleteOutlineIcon, EditIcon } from '@milesight/shared/src/components';
import { Tooltip, type ColumnType } from '@/components';
import { type TokenItemProps } from '@/services/http';

type OperationType = 'edit' | 'delete' | 'enable';

export type TableRowDataType = TokenItemProps;

export interface UseColumnsProps<T> {
    /**
     * operation Button callbacks
     */
    onButtonClick: (type: OperationType, record: T, otherProp?: unknown) => void;
}

const useColumns = <T extends TableRowDataType>({ onButtonClick }: UseColumnsProps<T>) => {
    const { getIntlText } = useI18n();
    const { getTimeFormat } = useTime();

    const columns: ColumnType<T>[] = useMemo(() => {
        return [
            {
                field: 'id',
                headerName: getIntlText('token.title.token_id'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
            },
            {
                field: 'name',
                headerName: getIntlText('token.title.token_name'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
            },
            {
                field: 'token',
                headerName: getIntlText('token.title.token_value'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
            },
            {
                field: 'remaining_requests',
                headerName: getIntlText('token.title.remaining_request_times'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
            },
            {
                field: 'status',
                headerName: getIntlText('common.label.enable_status'),
                align: 'left',
                headerAlign: 'left',
                type: 'boolean',
                filterable: true,
                disableColumnMenu: false,
                flex: 1,
                minWidth: 150,
                renderCell({ row }) {
                    return (
                        <Switch
                            checked={row.status === 'active'}
                            onChange={(_, checked) => onButtonClick('enable', row, checked)}
                        />
                    );
                },
            },
            {
                field: 'created_at',
                headerName: getIntlText('common.label.create_time'),
                flex: 1,
                minWidth: 150,
                ellipsis: true,
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
                renderCell({ value }) {
                    return getTimeFormat(Number(value));
                },
            },
            {
                field: '$operation',
                headerName: getIntlText('common.label.operation'),
                display: 'flex',
                width: 120,
                align: 'left',
                headerAlign: 'left',
                renderCell({ row }) {
                    return (
                        <Stack
                            direction="row"
                            spacing="4px"
                            sx={{ height: '100%', alignItems: 'center', justifyContent: 'end' }}
                        >
                            <Tooltip title={getIntlText('common.button.edit')}>
                                <IconButton
                                    sx={{ width: 30, height: 30 }}
                                    onClick={() => onButtonClick('edit', row)}
                                >
                                    <EditIcon sx={{ width: 20, height: 20 }} />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title={getIntlText('common.label.delete')}>
                                <IconButton
                                    sx={{
                                        width: 30,
                                        height: 30,
                                        color: 'text.secondary',
                                    }}
                                    onClick={() => onButtonClick('delete', row)}
                                >
                                    <DeleteOutlineIcon sx={{ width: 20, height: 20 }} />
                                </IconButton>
                            </Tooltip>
                        </Stack>
                    );
                },
            },
        ];
    }, [getIntlText, onButtonClick, getTimeFormat]);

    return columns;
};

export default useColumns;
