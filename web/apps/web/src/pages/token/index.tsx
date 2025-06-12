import React, { useState, useMemo } from 'react';
import { useMemoizedFn, useRequest } from 'ahooks';
import { Button, Stack } from '@mui/material';

import { useI18n } from '@milesight/shared/src/hooks';
import { AddIcon } from '@milesight/shared/src/components';
import { TablePro } from '@/components';

import { tokenAPI, awaitWrap, getResponseData, isRequestSuccess } from '@/services/http';
import {
    type UseColumnsProps,
    type TableRowDataType,
    useColumns,
    useTokenModal,
    useToken,
} from './hooks';
import { OperateTokenModal } from './components';
import { convertDataToDisplay } from './utils';

import styles from './style.module.less';

const Token: React.FC = () => {
    const { getIntlText } = useI18n();

    const [keyword, setKeyword] = useState<string>('');
    const [paginationModel, setPaginationModel] = useState({ page: 0, pageSize: 10 });

    const handleSearch = useMemoizedFn((value: string) => {
        setKeyword(value);
        setPaginationModel(model => ({ ...model, page: 0 }));
    });

    const {
        data: allTokens,
        loading,
        run: getAllTokens,
    } = useRequest(
        async () => {
            const { page, pageSize } = paginationModel;
            const [error, resp] = await awaitWrap(
                tokenAPI.getTokensList({
                    search: keyword,
                    page: page + 1,
                    page_size: pageSize,
                }),
            );
            if (error || !isRequestSuccess(resp)) {
                return;
            }

            const respData = getResponseData(resp);

            return {
                content: respData || [],
                total: resp?.data?.pagination?.total_items || 0,
            };
        },
        {
            debounceWait: 300,
            refreshDeps: [keyword, paginationModel],
        },
    );

    const { changeTokenEnableStatus, handleDeleteToken } = useToken(getAllTokens);

    const {
        tokenModalVisible,
        openAddToken,
        openEditToken,
        modalTitle,
        hideModal,
        onFormSubmit,
        operateType,
        currentToken,
    } = useTokenModal(getAllTokens);

    // ---------- Table render bar ----------
    const toolbarRender = useMemo(() => {
        return (
            <Stack className="ms-operations-btns" direction="row" spacing="12px">
                <Button
                    variant="contained"
                    className="md:d-none"
                    sx={{ height: 36, textTransform: 'none' }}
                    startIcon={<AddIcon />}
                    onClick={openAddToken}
                >
                    {getIntlText('common.label.add')}
                </Button>
            </Stack>
        );
    }, [getIntlText, openAddToken]);

    const handleTableBtnClick: UseColumnsProps<TableRowDataType>['onButtonClick'] = useMemoizedFn(
        (type, record, otherProps) => {
            switch (type) {
                case 'edit': {
                    openEditToken(record);
                    break;
                }
                case 'delete': {
                    handleDeleteToken(record);
                    break;
                }
                case 'enable': {
                    changeTokenEnableStatus(record, otherProps as boolean);
                    break;
                }
                default: {
                    break;
                }
            }
        },
    );

    const columns = useColumns<TableRowDataType>({ onButtonClick: handleTableBtnClick });

    return (
        <div className="ms-main">
            <div className={`ms-view ${styles['ms-view-token']}`}>
                <div className="ms-view__inner">
                    <div className={styles['token-wrapper']}>
                        <TablePro<TableRowDataType>
                            loading={loading}
                            columns={columns}
                            getRowId={row => row.id}
                            rows={allTokens?.content}
                            rowCount={allTokens?.total || 0}
                            paginationModel={paginationModel}
                            toolbarRender={toolbarRender}
                            onPaginationModelChange={setPaginationModel}
                            onSearch={handleSearch}
                            onRefreshButtonClick={getAllTokens}
                        />
                        {tokenModalVisible && (
                            <OperateTokenModal
                                data={currentToken ? convertDataToDisplay(currentToken) : undefined}
                                title={modalTitle}
                                operateType={operateType}
                                visible={tokenModalVisible}
                                onCancel={hideModal}
                                onFormSubmit={onFormSubmit}
                            />
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Token;
