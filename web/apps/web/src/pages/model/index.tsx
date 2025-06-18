import React, { useState, useMemo } from 'react';
import { useMemoizedFn, useRequest } from 'ahooks';
import { Button, Stack } from '@mui/material';

import { useI18n } from '@milesight/shared/src/hooks';
import { AddIcon } from '@milesight/shared/src/components';
import { TablePro } from '@/components';

import { modelAPI, awaitWrap, getResponseData, isRequestSuccess } from '@/services/http';
import {
    type UseColumnsProps,
    type TableRowDataType,
    useColumns,
    useModelModal,
    useModel,
} from './hooks';
import { OperateModelModal } from './components';

import styles from './style.module.less';

const Model: React.FC = () => {
    const { getIntlText } = useI18n();

    const [keyword, setKeyword] = useState<string>('');
    const [paginationModel, setPaginationModel] = useState({ page: 0, pageSize: 10 });
    const [sortType, setSortType] = useState<{ sortBy: string; sortOrder: SortType }>({
        sortBy: 'created_at',
        sortOrder: 'desc',
    });

    const handleSearch = useMemoizedFn((value: string) => {
        setKeyword(value);
        setPaginationModel(model => ({ ...model, page: 0 }));
    });

    const {
        data: allModels,
        loading,
        run: getAllModels,
    } = useRequest(
        async () => {
            const { page, pageSize } = paginationModel;
            const [error, resp] = await awaitWrap(
                modelAPI.getModelList({
                    search: keyword,
                    page: page + 1,
                    page_size: pageSize,
                    sort_by: sortType.sortBy,
                    sort_order: sortType.sortOrder,
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
            refreshDeps: [keyword, paginationModel, sortType],
        },
    );

    const { getInferEngines } = useModel();

    const {
        openAddModel,
        modelModalVisible,
        hideModal,
        operateType,
        modalTitle,
        onFormSubmit,
        currentModel,
    } = useModelModal(getAllModels, getInferEngines);

    // ---------- Table render bar ----------
    const toolbarRender = useMemo(() => {
        return (
            <Stack className="ms-operations-btns" direction="row" spacing="12px">
                <Button
                    variant="contained"
                    className="md:d-none"
                    sx={{ height: 36, textTransform: 'none' }}
                    startIcon={<AddIcon />}
                    onClick={openAddModel}
                >
                    {getIntlText('common.label.add')}
                </Button>
            </Stack>
        );
    }, [getIntlText, openAddModel]);

    const handleTableBtnClick: UseColumnsProps<TableRowDataType>['onButtonClick'] = useMemoizedFn(
        (type, record, otherProps) => {
            switch (type) {
                case 'edit': {
                    console.log('edit', record);
                    break;
                }
                case 'delete': {
                    console.log('delete', record);
                    break;
                }
                case 'enable': {
                    console.log('enable', record);
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
            <div className={`ms-view ${styles['ms-view-model']}`}>
                <div className="ms-view__inner">
                    <div className={styles['model-wrapper']}>
                        <TablePro<TableRowDataType>
                            loading={loading}
                            columns={columns}
                            getRowId={row => row.id}
                            rows={allModels?.content}
                            rowCount={allModels?.total || 0}
                            paginationModel={paginationModel}
                            toolbarRender={toolbarRender}
                            onPaginationModelChange={setPaginationModel}
                            onSearch={handleSearch}
                            onRefreshButtonClick={getAllModels}
                            sortingMode="server"
                            onSortModelChange={sorts => {
                                setSortType({
                                    sortBy: sorts?.[0]?.field || 'created_at',
                                    sortOrder: sorts?.[0]?.sort || 'desc',
                                });
                            }}
                        />
                        {modelModalVisible && (
                            <OperateModelModal
                                data={currentModel as unknown as any}
                                operateType={operateType}
                                title={modalTitle}
                                visible={modelModalVisible}
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

export default Model;
