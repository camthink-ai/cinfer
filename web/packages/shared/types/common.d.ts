/**
 * App type
 * @param web Web Application
 */
type AppType = 'web';

/**
 * Universal key type (Usually used for fields such as `id`, `key`, etc.)
 */
declare type ApiKey = string | number;

/**
 * Common API response type
 */
declare type ApiResponse<T = any> = {
    success: boolean;
    message: string;
    data?: T;
    error_code?: string;
    error_details?: any;
    pagination?: {
        current_page: number;
        page_size: number;
        total_items: number;
        total_pages: number;
    };
};

/**
 * API basic type definition
 */
declare type APISchema = Record<
    string,
    {
        request: Record<string, any> | any;
        response: Record<string, any> | any;
    }
>;

/**
 * Data sort type
 * @param ASC ascending order
 * @param DESC descending order
 */
declare type SortType = 'asc' | 'desc';

/**
 * Data sort props
 */
declare type SortsProps = {
    property?: string | number;
    direction?: SortType;
};

/**
 * Request data type for search API
 */
declare type SearchRequestType = {
    /** keyword to search */
    search?: string;

    /** Data count in single page */
    page_size?: number;

    /** Page number */
    page?: number;

    /** Data sort by key */
    sort_by?: string;

    /** Data sort order type */
    sort_order?: SortType;
};

/**
 * Response data type for search API
 */
declare type SearchResponseType<T = any[]> = {
    /** Data count in single page */
    page_size: number;
    /** Page number */
    page_number: number;
    /** total number */
    total: number;
    /** Paging list data */
    content: T;
};

/**
 * Map some fields in type T to optional fields, while keeping the other fields as they are
 */
declare type PartialOptional<T, K extends keyof T> = Omit<T, K> & {
    [P in K]?: T[P];
};

/**
 * Map some fields in type T to mandatory fields, while keeping the other fields as they are
 */
declare type RequiredOptional<T, K extends keyof T> = Omit<T, K> & {
    [P in K]-?: T[P];
};

/**
 * Convert underline to camel hump naming
 * @deprecated
 */
declare type SnakeToCamelCase<S extends string> = S extends `${infer T}_${infer U}`
    ? `${T}${Capitalize<SnakeToCamelCase<U>>}`
    : S;
/**
 * Recursive conversion of all attribute names in an object from underline naming to camel hump naming
 * @deprecated
 */
declare type ConvertKeysToCamelCase<T> = {
    [K in keyof T as SnakeToCamelCase<Extract<K, string>>]: T[K] extends object
        ? ConvertKeysToCamelCase<T[K]>
        : T[K];
};

/**
 * Convert string type from underline naming to camel hump naming
 */
declare type ToCamelCase<S extends string | number | symbol> = S extends string
    ? S extends `${infer Head}_${infer Tail}`
        ? `${ToCamelCase<Uncapitalize<Head>>}${Capitalize<ToCamelCase<Tail>>}`
        : Uncapitalize<S>
    : never;

/**
 * Recursive conversion of all attribute names in an object from
 * underline naming to camel hump naming
 */
declare type ObjectToCamelCase<T extends object | undefined | null> = T extends undefined
    ? undefined
    : T extends null
      ? null
      : T extends Array<infer ArrayType>
        ? ArrayType extends object
            ? Array<ObjectToCamelCase<ArrayType>>
            : Array<ArrayType>
        : T extends Uint8Array
          ? Uint8Array
          : T extends Date
            ? Date
            : {
                  [K in keyof T as ToCamelCase<K>]: T[K] extends
                      | Array<infer ArrayType>
                      | undefined
                      | null
                      ? ArrayType extends object
                          ? Array<ObjectToCamelCase<ArrayType>>
                          : Array<ArrayType>
                      : T[K] extends object | undefined | null
                        ? ObjectToCamelCase<T[K]>
                        : T[K];
              };

declare interface OptionsProps<T extends string | number = string | number> {
    label: string;
    value?: T;
    options?: OptionsProps<T>[];
}

declare interface SelectOptionsProps<T extends string | number = string | number> {
    label: string;
    value: T;
}

/**
 * forwardRef definition hack
 *
 * Inspired by: https://fettblog.eu/typescript-react-generic-forward-refs/
 */
declare type FixedForwardRef = <T, P = object>(
    render: (props: P, ref: React.Ref<T>) => React.ReactNode,
) => (props: P & React.RefAttributes<T>) => React.ReactNode;

declare type HttpMethodType =
    | 'GET'
    | 'POST'
    | 'PUT'
    | 'DELETE'
    | 'HEAD'
    | 'OPTIONS'
    | 'CONNECT'
    | 'TRACE'
    | 'PATCH';
