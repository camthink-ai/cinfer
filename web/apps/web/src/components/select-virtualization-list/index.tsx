import React, { useEffect } from 'react';
import { VariableSizeList, ListChildComponentProps } from 'react-window';
import {
    ListSubheader,
    Typography,
    type AutocompleteRenderOptionState,
    Checkbox,
} from '@mui/material';

import { CheckBoxOutlineBlankIcon, CheckBoxIcon } from '@milesight/shared/src/components';
import Tooltip from '../tooltip';

export interface SelectVirtualizationListProps {
    children?: React.ReactNode;
}

const LISTBOX_PADDING = 8; // px

function renderRow(props: ListChildComponentProps) {
    const { data, index, style } = props;
    const dataSet = data[index];
    const inlineStyle = {
        ...style,
        top: (style.top as number) + LISTBOX_PADDING,
    };

    if (Reflect.has(dataSet, 'group')) {
        return (
            <ListSubheader key={dataSet.key} component="div" style={inlineStyle}>
                {dataSet.group}
            </ListSubheader>
        );
    }

    const { key, ...optionProps } = dataSet[0];
    const option = dataSet[1] as OptionsProps;
    const state = dataSet[2] as AutocompleteRenderOptionState & { multiple: boolean };
    const optionLabel = option?.label || '';

    return (
        <Typography
            key={key}
            component="div"
            {...optionProps}
            noWrap
            style={inlineStyle}
            sx={{
                '.ms-tooltip': {
                    flex: 1,
                    width: 0,
                },
            }}
        >
            {state?.multiple && (
                <Checkbox
                    icon={<CheckBoxOutlineBlankIcon fontSize="small" />}
                    checkedIcon={<CheckBoxIcon fontSize="small" />}
                    style={{ marginRight: 8 }}
                    checked={state.selected}
                />
            )}
            <Tooltip autoEllipsis title={optionLabel}>
                <span className="ms-tooltip-cont">{optionLabel}</span>
            </Tooltip>
        </Typography>
    );
}

const OuterElementContext = React.createContext({});

const OuterElementType = React.forwardRef<HTMLDivElement>((props, ref) => {
    const outerProps = React.useContext(OuterElementContext);
    return <div ref={ref} {...props} {...outerProps} />;
});

function useResetCache(data: any) {
    const ref = React.useRef<VariableSizeList>(null);

    useEffect(() => {
        if (ref.current != null) {
            ref.current.resetAfterIndex(0, true);
        }
    }, [data]);
    return ref;
}

/**
 * The list is virtualized by react-window
 */
const SelectVirtualizationList = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLElement>
>(function ListboxComponent(props, ref) {
    const { children, ...other } = props;
    const itemData: React.ReactElement<unknown>[] = [];
    (children as React.ReactElement<unknown>[]).forEach(
        (
            item: React.ReactElement<unknown> & {
                children?: React.ReactElement<unknown>[];
            },
        ) => {
            itemData.push(item);
            itemData.push(...(item.children || []));
        },
    );

    const itemCount = itemData.length;
    const itemSize = 36;

    const getChildSize = (child: React.ReactElement<unknown>) => {
        if (Reflect.has(child, 'group')) {
            return 48;
        }

        return itemSize;
    };

    const getHeight = () => {
        if (itemCount > 8) {
            return 8 * itemSize;
        }
        return itemData.map(getChildSize).reduce((a, b) => a + b, 0);
    };

    const gridRef = useResetCache(itemCount);

    return (
        <div ref={ref}>
            <OuterElementContext.Provider value={other}>
                <VariableSizeList
                    itemData={itemData}
                    height={getHeight() + 2 * LISTBOX_PADDING}
                    width="100%"
                    ref={gridRef}
                    outerElementType={OuterElementType}
                    innerElementType="div"
                    itemSize={index => getChildSize(itemData[index])}
                    overscanCount={5}
                    itemCount={itemCount}
                >
                    {renderRow}
                </VariableSizeList>
            </OuterElementContext.Provider>
        </div>
    );
});

export default SelectVirtualizationList;
